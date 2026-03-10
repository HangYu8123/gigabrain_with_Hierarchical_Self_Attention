"""Hierarchical Self-Attention (HSA) for GigaBrain-0.

Implements the Hierarchical Self-Attention mechanism from
"Hierarchical Self-Attention: Generalizing Neural Attention Mechanics
to Multi-Scale Problems" (arXiv:2509.15448, NeurIPS 2025).

The key idea: instead of flat self-attention over all multi-modal tokens
(vision patches from 3 cameras + language tokens + action tokens),
structure the tokens into a hierarchy:

    Root (entire observation)
    ├── Language group
    ├── Vision group
    │   ├── cam_high patches
    │   ├── cam_left_wrist patches
    │   └── cam_right_wrist patches
    └── Action group

HSA computes attention respecting this hierarchy via a block-constrained
attention matrix that is provably the closest to flat Softmax attention
while encoding the hierarchical inductive bias (Theorem 3.2 in the paper).

Two components are provided:
1. HierarchicalSelfAttention: Standalone HSA module implementing the
   paper's dynamic programming algorithm (Algorithms 1-3).
2. HierarchicalAttentionBias: Learnable block-structured attention bias
   that can be injected into existing transformer layers via hooks,
   compatible with torch.compile, FSDP, and activation checkpointing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Large negative constant used instead of float('inf') for leaf phi values.
# Using -inf would produce NaN gradients via logsumexp; this approximation
# ensures exp(_NEG_INF_APPROX) ≈ 0 without breaking gradient flow.
_NEG_INF_APPROX = 1e4


# ---------------------------------------------------------------------------
# Hierarchy data structures
# ---------------------------------------------------------------------------

@dataclass
class HierarchyNode:
    """A node in the signal hierarchy tree.

    Leaf nodes represent contiguous token ranges [start, end) in the
    flattened sequence. Internal nodes aggregate their children.
    """

    name: str
    start: int  # inclusive index into the token sequence
    end: int  # exclusive index into the token sequence
    children: list[HierarchyNode] = field(default_factory=list)
    depth: int = 0

    @property
    def num_leaves(self) -> int:
        return self.end - self.start

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def build_signal_hierarchy(
    token_boundaries: dict[str, tuple[int, int]],
    total_seq_len: int,
) -> HierarchyNode:
    """Build a signal hierarchy tree from token group boundaries.

    Args:
        token_boundaries: Maps group names to (start, end) indices.
            Expected keys: 'lang', 'cam_high', 'cam_left_wrist',
            'cam_right_wrist', 'action'. Values are 0-based indices
            into the flattened token sequence.
        total_seq_len: Total sequence length (sum of all token groups).

    Returns:
        The root HierarchyNode of the signal hierarchy.
    """
    lang_start, lang_end = token_boundaries['lang']
    cam_high_start, cam_high_end = token_boundaries['cam_high']
    cam_left_start, cam_left_end = token_boundaries['cam_left_wrist']
    cam_right_start, cam_right_end = token_boundaries['cam_right_wrist']
    action_start, action_end = token_boundaries['action']

    # Leaf-parent nodes for language and action
    lang_node = HierarchyNode(
        name='language', start=lang_start, end=lang_end, depth=2,
    )
    action_node = HierarchyNode(
        name='action', start=action_start, end=action_end, depth=2,
    )

    # Camera leaf nodes under the vision internal node
    cam_high_node = HierarchyNode(
        name='cam_high', start=cam_high_start, end=cam_high_end, depth=2,
    )
    cam_left_node = HierarchyNode(
        name='cam_left_wrist', start=cam_left_start, end=cam_left_end, depth=2,
    )
    cam_right_node = HierarchyNode(
        name='cam_right_wrist', start=cam_right_start, end=cam_right_end, depth=2,
    )

    # Vision internal node (depth 1)
    vision_node = HierarchyNode(
        name='vision',
        start=min(cam_high_start, cam_left_start, cam_right_start),
        end=max(cam_high_end, cam_left_end, cam_right_end),
        children=[cam_high_node, cam_left_node, cam_right_node],
        depth=1,
    )

    # Root node (depth 0)
    root = HierarchyNode(
        name='root',
        start=0,
        end=total_seq_len,
        children=[lang_node, vision_node, action_node],
        depth=0,
    )

    return root


# ---------------------------------------------------------------------------
# Core HSA module — Paper Algorithms 1-3
# ---------------------------------------------------------------------------

class HierarchicalSelfAttention(nn.Module):
    """Implements HSA from arXiv:2509.15448 (Algorithms 1-3).

    Given a signal hierarchy tree and query/key/value tensors, computes
    hierarchical self-attention via dynamic programming:
    1. Bottom-up: compute per-node sufficient statistics
    2. Top-down: compute hierarchical attention output

    This is a standalone module that can be applied to any token sequence
    organized into a signal hierarchy. It does NOT require modifying
    any model internals.
    """

    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        hierarchy: HierarchyNode,
    ) -> torch.Tensor:
        """Apply HSA to token sequence x according to hierarchy.

        Args:
            x: Token embeddings [B, S, D].
            hierarchy: Signal hierarchy tree over the sequence dimension.

        Returns:
            Updated token embeddings [B, S, D] after HSA.
        """
        B, S, D = x.shape
        residual = x
        x_norm = self.layer_norm(x)

        Q = self.q_proj(x_norm)  # [B, S, D]
        K = self.k_proj(x_norm)  # [B, S, D]
        V = self.v_proj(x_norm)  # [B, S, D]

        # Reshape for multi-head: [B, num_heads, S, head_dim]
        Q = Q.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Normalize Q and K (as required by HSA derivation — Proposition 3.1)
        Q = F.normalize(Q, dim=-1) * math.sqrt(self.head_dim)
        K = F.normalize(K, dim=-1) * math.sqrt(self.head_dim)

        # Bottom-up: compute sufficient statistics per node
        node_stats = self._bottom_up(Q, K, V, hierarchy)

        # Top-down: compute hierarchical attention output
        output = self._top_down(Q, K, V, hierarchy, node_stats)

        # output: [B, num_heads, S, head_dim] -> [B, S, D]
        output = output.transpose(1, 2).reshape(B, S, D)
        output = self.o_proj(output)

        return residual + output

    def _bottom_up(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        node: HierarchyNode,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Bottom-up pass: compute sufficient statistics per node.

        For each node A, computes:
        - rho_q(A): mean query vector over leaves of A
        - rho_k(A): mean key vector over leaves of A
        - mu_v(A): mean value vector over leaves of A
        - phi(A): energy of the sub-hierarchy rooted at A
        - eta(A): cross-sibling interaction summary

        Args:
            Q, K, V: [B, H, S, d] tensors.
            node: Current node in the hierarchy.

        Returns:
            Dictionary mapping node names to their statistics.
        """
        stats: dict[str, dict[str, torch.Tensor]] = {}
        self._compute_stats_recursive(Q, K, V, node, stats)
        return stats

    def _compute_stats_recursive(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        node: HierarchyNode,
        stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Recursively compute sufficient statistics (Algorithm 2)."""
        s, e = node.start, node.end
        n = e - s  # |ell(A)|

        # Mean query and key for this node (Eq. 42)
        rho_q = Q[:, :, s:e, :].mean(dim=2)  # [B, H, d]
        rho_k = K[:, :, s:e, :].mean(dim=2)  # [B, H, d]
        mu_v = V[:, :, s:e, :].mean(dim=2)  # [B, H, d]

        node_stat: dict[str, Any] = {
            'rho_q': rho_q,
            'rho_k': rho_k,
            'mu_v': mu_v,
            'num_leaves': n,
            'start': s,
            'end': e,
        }

        if node.is_leaf or len(node.children) == 0:
            # Leaf energy: use large positive value so exp(-phi) ≈ 0.
            # Always [B, H] shaped to maintain gradient flow.
            node_stat['phi'] = torch.full(
                (Q.shape[0], Q.shape[1]), _NEG_INF_APPROX,
                device=Q.device, dtype=Q.dtype,
            )
        else:
            # Recurse on children first
            for child in node.children:
                self._compute_stats_recursive(Q, K, V, child, stats)

            # Compute energy phi(A) via Eq. 8
            # phi(A) = -sum_B (|l(B)|/|l(A)|) * log[exp(-phi(B)) + sum_C |l(C)|*exp(-psi_{B->C})]
            d = Q.shape[-1]
            phi_val = torch.zeros(Q.shape[0], Q.shape[1], device=Q.device)  # [B, H]

            for child in node.children:
                child_stats = stats[child.name]
                child_phi = child_stats['phi']  # [B, H] (always batched)
                n_child = child_stats['num_leaves']
                child_rho_q = child_stats['rho_q']  # [B, H, d]
                child_rho_k = child_stats['rho_k']  # [B, H, d]

                # exp(-phi(B)): alpha term — phi is always [B, H]
                alpha_log = -child_phi  # [B, H]

                # sum_C |l(C)| * exp(-psi_{B->C}) for siblings C of child
                sibling_log_terms = []
                for sibling in node.children:
                    if sibling.name == child.name:
                        continue
                    sib_stats = stats[sibling.name]
                    n_sib = sib_stats['num_leaves']

                    # psi_{B->C} from Eq. 16 (with LayerNorm):
                    # = d - (1/(d * |l(B)| * |l(C)|)) sum_i sum_j q_i^T k_j
                    # = d - rho_q(B)^T rho_k(C)  (since average dotproduct = mean_q dot mean_k * |l(B)|*|l(C)| / (|l(B)|*|l(C)|))
                    # Simplified: psi = d - rho_q(B)^T rho_k(C) / d
                    psi = d - (child_rho_q * sib_stats['rho_k']).sum(dim=-1) / d  # [B, H]

                    # log(|l(C)| * exp(-psi)) = log|l(C)| - psi
                    log_term = math.log(max(n_sib, 1)) - psi
                    sibling_log_terms.append(log_term)

                if len(sibling_log_terms) > 0:
                    # log[exp(-phi(B)) + sum_C |l(C)|*exp(-psi_{B->C})]
                    all_log_terms = [alpha_log] + sibling_log_terms
                    stacked = torch.stack(all_log_terms, dim=-1)  # [B, H, K]
                    log_sum = torch.logsumexp(stacked, dim=-1)  # [B, H]
                else:
                    log_sum = alpha_log

                weight = n_child / max(n, 1)
                phi_val = phi_val - weight * log_sum

            node_stat['phi'] = phi_val

        stats[node.name] = node_stat

    def _top_down(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        root: HierarchyNode,
        stats: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Top-down pass: compute hierarchical attention output (Algorithm 3).

        Instead of directly computing gradient vectors, we compute
        hierarchical attention weights and apply them to values.

        Args:
            Q, K, V: [B, H, S, d] tensors.
            root: Root of the signal hierarchy.
            stats: Node statistics from bottom-up pass.

        Returns:
            Attention output [B, H, S, d].
        """
        B, H, S, d = Q.shape
        output = torch.zeros_like(Q)  # [B, H, S, d]

        # For each leaf group, compute attention output
        self._compute_attention_recursive(
            Q, K, V, root, root, stats, output,
        )

        return output

    def _compute_attention_recursive(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        node: HierarchyNode,
        root: HierarchyNode,
        stats: dict[str, dict[str, torch.Tensor]],
        output: torch.Tensor,
    ) -> None:
        """Recursively compute attention for each leaf group.

        For each leaf group (leaf-parent node A), we compute:
        1. Intra-group attention: standard softmax within leaves of A
        2. Inter-group attention: weighted contribution from sibling
           groups according to the HSA block weights

        These are combined using mu (intra weight) and delta (inter weights)
        derived from the energies and interaction energies (Eq. 19-20).
        """
        if node.is_leaf or len(node.children) == 0:
            # At a leaf-parent: compute local attention + aggregated sibling contributions
            return

        d = Q.shape[-1]
        parent_stats = stats[node.name]

        for child in node.children:
            child_stats = stats[child.name]
            child_phi = child_stats['phi']  # [B, H] (always batched)
            n_child = child_stats['num_leaves']
            cs, ce = child_stats['start'], child_stats['end']

            # Compute alpha(B) = exp(-phi(B)) — phi is always [B, H]
            log_alpha = -child_phi  # [B, H]

            # Compute denominator: alpha(B) + sum_C |l(C)| * beta(B,C)
            sibling_terms = []
            sibling_info = []
            for sibling in node.children:
                if sibling.name == child.name:
                    continue
                sib_stats = stats[sibling.name]
                n_sib = sib_stats['num_leaves']

                psi = d - (child_stats['rho_q'] * sib_stats['rho_k']).sum(dim=-1) / d
                log_beta = -psi  # [B, H]
                log_weighted = math.log(max(n_sib, 1)) + log_beta
                sibling_terms.append(log_weighted)
                sibling_info.append((sibling, sib_stats, log_beta))

            if len(sibling_terms) > 0:
                all_terms = [log_alpha] + sibling_terms
                stacked = torch.stack(all_terms, dim=-1)  # [B, H, K+1]
                log_denom = torch.logsumexp(stacked, dim=-1)  # [B, H]
            else:
                log_denom = log_alpha

            # mu(B) = alpha(B) / denom  (Eq. 19)
            log_mu = log_alpha - log_denom  # [B, H]
            mu = torch.exp(log_mu)  # [B, H]

            if child.is_leaf or len(child.children) == 0:
                # Leaf-parent: compute intra-group softmax attention
                Q_local = Q[:, :, cs:ce, :]  # [B, H, n_child, d]
                K_local = K[:, :, cs:ce, :]
                V_local = V[:, :, cs:ce, :]

                # Standard scaled dot-product attention within the group
                attn_scores = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(d)
                attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, n_child, n_child]
                intra_out = torch.matmul(attn_weights, V_local)  # [B, H, n_child, d]

                # Combine: mu * intra + sum_C delta(B,C) * mu_v(C)
                combined = mu.unsqueeze(-1).unsqueeze(-1) * intra_out  # [B, H, n_child, d]

                for sibling, sib_stats, log_beta in sibling_info:
                    n_sib = sib_stats['num_leaves']
                    # delta(B,C) = beta(B,C) / denom  (Eq. 20)
                    log_delta = log_beta - log_denom
                    delta = torch.exp(log_delta)  # [B, H]

                    # Contribution from sibling: delta * |l(C)| * mean_v(C)
                    # But we expand to all tokens in child
                    sib_v_mean = sib_stats['mu_v']  # [B, H, d]
                    contrib = (delta * n_sib).unsqueeze(-1).unsqueeze(-1) * sib_v_mean.unsqueeze(-2).expand(intra_out.shape)
                    combined = combined + contrib

                output[:, :, cs:ce, :] = combined
            else:
                # Internal node: recurse into children
                self._compute_attention_recursive(
                    Q, K, V, child, root, stats, output,
                )

                # After recursion, blend with sibling contributions
                for sibling, sib_stats, log_beta in sibling_info:
                    n_sib = sib_stats['num_leaves']
                    log_delta = log_beta - log_denom
                    delta = torch.exp(log_delta)

                    sib_v_mean = sib_stats['mu_v']
                    # Add sibling contribution to all tokens in this child
                    child_slice = output[:, :, cs:ce, :]
                    contrib = (delta * n_sib).unsqueeze(-1).unsqueeze(-1) * sib_v_mean.unsqueeze(-2).expand(
                        child_slice.shape,
                    )
                    output[:, :, cs:ce, :] = mu.unsqueeze(-1).unsqueeze(-1) * child_slice + contrib


# ---------------------------------------------------------------------------
# Hierarchical Attention Bias — torch.compile compatible
# ---------------------------------------------------------------------------

class HierarchicalAttentionBias(nn.Module):
    """Learnable block-structured attention bias encoding the signal hierarchy.

    Produces a [seq_len x seq_len] bias matrix B where:
        B[i,j] = b_group(group(i), group(j))

    This bias is added to attention scores before softmax:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d) + B) V

    The block structure naturally focuses attention within semantically
    related groups (e.g., same camera, same modality) while allowing
    controlled cross-group interaction.

    Initialized at zero for safe passthrough from pre-trained weights.
    """

    def __init__(
        self,
        num_groups: int,
        seq_len: int,
        group_assignments: torch.Tensor,
    ):
        """Initialize the hierarchical attention bias.

        Args:
            num_groups: Number of token groups in the hierarchy.
            seq_len: Total sequence length.
            group_assignments: [seq_len] tensor mapping each token to its
                group index (0-indexed).
        """
        super().__init__()
        self.num_groups = num_groups
        self.seq_len = seq_len

        # Register group assignments as buffer (not a parameter)
        self.register_buffer('group_assignments', group_assignments)

        # Learnable inter-group bias: [num_groups, num_groups]
        # Initialized to zero → no bias initially (safe passthrough)
        self.group_bias = nn.Parameter(torch.zeros(num_groups, num_groups))

    def forward(self) -> torch.Tensor:
        """Compute the full attention bias matrix.

        Returns:
            Bias matrix [seq_len, seq_len].
        """
        # Index into the group_bias using group assignments
        # group_assignments: [S], group_bias: [G, G] -> bias: [S, S]
        row_groups = self.group_assignments.unsqueeze(1).expand(-1, self.seq_len)   # [S, S]
        col_groups = self.group_assignments.unsqueeze(0).expand(self.seq_len, -1)   # [S, S]

        bias = self.group_bias[row_groups, col_groups]  # [S, S]
        return bias


# ---------------------------------------------------------------------------
# HSA Pre-processor: applies HSA before the main model
# ---------------------------------------------------------------------------

class HSAPreProcessor(nn.Module):
    """Pre-processes multi-modal token embeddings with HSA.

    This module sits between the embedding/projection layers and the
    main transformer decoder. It applies one layer of Hierarchical
    Self-Attention to the concatenated multi-modal token sequence,
    narrowing attention focus to task-relevant objects before the
    main model processes the features.

    Compatible with torch.compile when hierarchy_meta is static.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_groups: int = 5,
    ):
        """Initialize the HSA pre-processor.

        Args:
            d_model: Model hidden dimension.
            num_heads: Number of attention heads for HSA.
            num_groups: Number of groups in the hierarchy
                (default 5: lang, cam_high, cam_left, cam_right, action).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.hsa = HierarchicalSelfAttention(d_model, num_heads)

        # Learnable gate controlling HSA contribution (initialized near zero)
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        x: torch.Tensor,
        hierarchy: HierarchyNode,
    ) -> torch.Tensor:
        """Apply gated HSA to token embeddings.

        Args:
            x: Token embeddings [B, S, D].
            hierarchy: Signal hierarchy tree.

        Returns:
            Updated embeddings [B, S, D]. With gate≈0 at init,
            this is approximately identity.
        """
        hsa_out = self.hsa(x, hierarchy)
        gate = self.gate
        return (1 - gate) * x + gate * hsa_out


# ---------------------------------------------------------------------------
# Utility: build hierarchy metadata for GigaBrain-0
# ---------------------------------------------------------------------------

def build_gigabrain_hierarchy_meta(
    num_vision_tokens_per_camera: int = 256,
    num_cameras: int = 3,
    max_lang_length: int = 200,
    action_chunk: int = 50,
    vision_first: bool = True,
) -> dict[str, Any]:
    """Build hierarchy metadata for GigaBrain-0's token layout.

    The GigaBrain-0 model (PaliGemma2-based) typically has the token
    sequence ordered as: [vision tokens | language tokens | action tokens].

    Vision tokens: num_cameras * num_vision_tokens_per_camera
    Language tokens: max_lang_length
    Action tokens: action_chunk

    Args:
        num_vision_tokens_per_camera: Number of vision patch tokens per
            camera view. For 224x224 images with 14x14 patches → 256.
        num_cameras: Number of camera views (default 3).
        max_lang_length: Maximum language token sequence length.
        action_chunk: Number of action chunk tokens.
        vision_first: If True, vision tokens come first; if False,
            language tokens come first.

    Returns:
        Dictionary with:
        - 'token_boundaries': mapping group names to (start, end)
        - 'total_seq_len': total sequence length
        - 'hierarchy': the built HierarchyNode tree
        - 'group_assignments': [total_seq_len] tensor of group indices
    """
    n_vision = num_cameras * num_vision_tokens_per_camera

    if vision_first:
        # [vision | language | action]
        cam_high_start = 0
        cam_high_end = num_vision_tokens_per_camera
        cam_left_start = num_vision_tokens_per_camera
        cam_left_end = 2 * num_vision_tokens_per_camera
        cam_right_start = 2 * num_vision_tokens_per_camera
        cam_right_end = 3 * num_vision_tokens_per_camera
        lang_start = n_vision
        lang_end = n_vision + max_lang_length
        action_start = n_vision + max_lang_length
        action_end = n_vision + max_lang_length + action_chunk
    else:
        # [language | vision | action]
        lang_start = 0
        lang_end = max_lang_length
        cam_high_start = max_lang_length
        cam_high_end = max_lang_length + num_vision_tokens_per_camera
        cam_left_start = max_lang_length + num_vision_tokens_per_camera
        cam_left_end = max_lang_length + 2 * num_vision_tokens_per_camera
        cam_right_start = max_lang_length + 2 * num_vision_tokens_per_camera
        cam_right_end = max_lang_length + 3 * num_vision_tokens_per_camera
        action_start = max_lang_length + n_vision
        action_end = max_lang_length + n_vision + action_chunk

    total_seq_len = n_vision + max_lang_length + action_chunk

    token_boundaries = {
        'lang': (lang_start, lang_end),
        'cam_high': (cam_high_start, cam_high_end),
        'cam_left_wrist': (cam_left_start, cam_left_end),
        'cam_right_wrist': (cam_right_start, cam_right_end),
        'action': (action_start, action_end),
    }

    hierarchy = build_signal_hierarchy(token_boundaries, total_seq_len)

    # Group assignments for HierarchicalAttentionBias
    # Groups: 0=lang, 1=cam_high, 2=cam_left, 3=cam_right, 4=action
    group_assignments = torch.zeros(total_seq_len, dtype=torch.long)
    group_assignments[lang_start:lang_end] = 0
    group_assignments[cam_high_start:cam_high_end] = 1
    group_assignments[cam_left_start:cam_left_end] = 2
    group_assignments[cam_right_start:cam_right_end] = 3
    group_assignments[action_start:action_end] = 4

    return {
        'token_boundaries': token_boundaries,
        'total_seq_len': total_seq_len,
        'hierarchy': hierarchy,
        'group_assignments': group_assignments,
    }


# ---------------------------------------------------------------------------
# Model integration: apply HSA bias to GigaBrain0Policy
# ---------------------------------------------------------------------------

def apply_hsa_bias_hooks(
    model: nn.Module,
    attention_bias: HierarchicalAttentionBias,
    target_layer_indices: list[int] | None = None,
) -> list[torch.utils.hooks.RemovableHook]:
    """Attach HSA bias hooks to target attention layers.

    Searches for attention layers in the model and registers forward
    pre-hooks that add the hierarchical attention bias to the attention
    mask (which in HuggingFace Gemma2 is an additive float mask applied
    to attention scores before softmax).

    Args:
        model: The GigaBrain0Policy model.
        attention_bias: The hierarchical attention bias module.
        target_layer_indices: Which decoder layer indices to apply
            HSA bias to. If None, applies to all layers.

    Returns:
        List of hook handles (for later removal if needed).

    Raises:
        RuntimeError: If no attention layers are found.
    """
    hooks = []
    layer_count = 0
    found_attn_names = []

    for name, module in model.named_modules():
        # Match attention layers with various naming conventions:
        # - paligemma_with_expert.model.layers.N.self_attn
        # - model.layers.N.self_attn
        # - layers.N.self_attn
        if 'self_attn' not in name:
            continue

        # Extract layer index from "layers.N" pattern anywhere in the name
        parts = name.split('.')
        layer_idx = None
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        if layer_idx is None:
            continue

        found_attn_names.append(f'{name} (layer {layer_idx})')

        if target_layer_indices is None or layer_idx in target_layer_indices:
            hook = _register_bias_hook(module, attention_bias)
            hooks.append(hook)
            layer_count += 1

    if layer_count == 0:
        all_module_names = [n for n, _ in model.named_modules() if 'attn' in n.lower()]
        raise RuntimeError(
            f'HSA bias hooks: no attention layers found to hook. '
            f'Searched for "self_attn" modules but found none matching. '
            f'Attention-related modules in model: {all_module_names[:20]}',
        )

    logger.info(
        'HSA bias hooks: attached to %d attention layers. '
        'Found layers: %s', layer_count, found_attn_names[:5],
    )

    if target_layer_indices is not None and layer_count != len(target_layer_indices):
        import warnings
        warnings.warn(
            f'HSA bias hooks: expected {len(target_layer_indices)} layers, '
            f'found {layer_count}. Available attention modules may have '
            f'different naming convention.',
            stacklevel=2,
        )

    return hooks


def _register_bias_hook(
    attn_module: nn.Module,
    attention_bias: HierarchicalAttentionBias,
) -> torch.utils.hooks.RemovableHook:
    """Register a forward pre-hook that adds HSA bias to attention scores.

    In HuggingFace Gemma2 (used by PaliGemma2), `attention_mask` is
    an additive float mask (with -inf/0 values) that gets added to
    attention scores before softmax: ``scores = QK^T / sqrt(d) + mask``.
    Adding the learnable HSA bias to this mask effectively injects
    hierarchical structure into the attention computation.
    """
    def hook_fn(module, args, kwargs):
        if 'attention_mask' not in kwargs or kwargs['attention_mask'] is None:
            return args, kwargs

        mask = kwargs['attention_mask']
        bias_matrix = attention_bias()  # [S, S]
        # Move to same device/dtype as mask
        bias_matrix = bias_matrix.to(device=mask.device, dtype=mask.dtype)

        # attention_mask is typically [B, 1, S, S] or [B, H, S, S]
        bias_expanded = bias_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Trim or pad bias to match mask size
        mask_s = mask.shape[-1]
        bias_s = bias_expanded.shape[-1]
        if bias_s > mask_s:
            bias_expanded = bias_expanded[:, :, :mask_s, :mask_s]
        elif bias_s < mask_s:
            pad = torch.zeros(
                1, 1, mask_s, mask_s,
                device=mask.device, dtype=mask.dtype,
            )
            pad[:, :, :bias_s, :bias_s] = bias_expanded
            bias_expanded = pad

        kwargs = dict(kwargs)
        kwargs['attention_mask'] = mask + bias_expanded

        return args, kwargs

    return attn_module.register_forward_pre_hook(hook_fn, with_kwargs=True)
