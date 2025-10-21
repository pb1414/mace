###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_mean, scatter_sum, scatter_max, scatter_softmax
from mace.tools.torch_tools import get_change_of_basis, spherical_to_cartesian
import time

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)

import torch
from torch import nn

class GaussianRBF(nn.Module):
    def __init__(self, num_centers: int, r_min: float = 0.0, r_max: float = 1.0):
        super().__init__()
        centers = torch.linspace(r_min, r_max, num_centers)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff ** 2)


class RangeRelay(nn.Module):
    """
    Vectorized single-master RANGE with parallel head processing.

    - Projects Q/K/V/E for all heads in a single linear for nodes and for the master template.
    - Uses per-head weight tensors for master-side linear maps (B_K, B_V) but applies them
      via batch matmul (no Python loop).
    - Uses flattening trick to call your scatter_softmax/scatter_sum once per needed reduction
      (index = batch * num_heads + head_id).
    """

    def __init__(
        self,
        in_dim: int,
        head_dim: int,
        num_heads: int,
        rbf_dim: int = 7,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert in_dim % num_heads == 0 or True, "in_dim no longer required to be divisible by num_heads"
        self.h = in_dim
        self.d = head_dim
        self.num_heads = num_heads
        self.L = num_heads
        self.leaky = nn.LeakyReLU(negative_slope)

        # RBF
        self.rbf = GaussianRBF(rbf_dim)

        # Single master embedding per graph (template)
        self.H0 = nn.Parameter(torch.randn(1, self.h))

        # ---- Node-side joint projections (one linear each that outputs all heads) ----
        # These map node scalar features -> [num_heads * head_dim], then we reshape to [N, num_heads, d]
        self.A_K = nn.Linear(self.h, self.num_heads * self.d, bias=False)
        self.A_V = nn.Linear(self.h, self.num_heads * self.d, bias=False)
        self.A_Q_master = nn.Linear(self.h, self.num_heads * self.d, bias=False)  # applied to H0
        self.A_E = nn.Linear(rbf_dim, self.num_heads * self.d, bias=False)

        # per-head attention vectors (a^l) packed as [num_heads, d]
        self.a_vec = nn.Parameter(torch.randn(self.num_heads, self.d))

        # ---- Broadcast-side node projections (node -> per-head) ----
        # These are applied to nodes and produce [N, num_heads, d]
        self.B_Q_node = nn.Linear(self.h, self.num_heads * self.d, bias=False)
        self.B_K_node = nn.Linear(self.h, self.num_heads * self.d, bias=False)
        self.B_E = nn.Linear(rbf_dim, self.num_heads * self.d, bias=False)
        self.B_V_self = nn.Linear(self.h, self.num_heads * self.d, bias=False)

        # per-head scalar b_vec
        self.b_vec = nn.Parameter(torch.randn(self.num_heads, self.d))

        # ---- Master-side per-head linear maps (batched matrices) ----
        # We'll store per-head weight matrices for mapping master-head -> head-key/value.
        # Shapes: [num_heads, d, d]
        self.BK_master_weight = nn.Parameter(torch.randn(self.num_heads, self.d, self.d) * np.sqrt(1.0 / self.d))
        self.BK_master_bias = nn.Parameter(torch.zeros(self.num_heads, self.d))
        self.BV_master_weight = nn.Parameter(torch.randn(self.num_heads, self.d, self.d) * np.sqrt(1.0 / self.d))
        self.BV_master_bias = nn.Parameter(torch.zeros(self.num_heads, self.d))

        # small projection from concatenated heads -> H
        self.to_H = nn.Linear(self.num_heads * self.d, self.h, bias=False)

        # Mix heads -> back to in_dim (final residual branch)
        hidden = 192 #max(128, self.num_heads * self.d // 2)
        self.mix = nn.Sequential(
            nn.Linear(self.num_heads * self.d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

        # gating scalar (optional, useful to start small)
        # self.alpha = nn.Parameter(torch.tensor(0.01))

        # sensible small init: scaled xavier for A_* and B_*; small a/b vectors
        # with torch.no_grad():
        #     nn.init.xavier_uniform_(self.A_K.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.A_V.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.A_Q_master.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.A_E.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.B_Q_node.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.B_K_node.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.B_V_self.weight, gain=1.0)
        #     nn.init.xavier_uniform_(self.B_E.weight, gain=1.0)
        #     # small attention scalars to start mild
        #     self.a_vec.mul_(0.01)
        #     self.b_vec.mul_(0.01)
        #     # final mix small
        #     nn.init.xavier_uniform_(self.mix[0].weight, gain=0.5)
        #     nn.init.zeros_(self.mix[-1].weight)
        #     if self.mix[-1].bias is not None:
        #         nn.init.zeros_(self.mix[-1].bias)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        import time
        t0 = time.time()
        N = h.shape[0]

        # ---------- per-graph centers & normalized radial encoding ----------
        centers = scatter_mean(pos, batch, dim=0)  # [G,3]
        t1 = time.time()
        d = torch.linalg.norm(pos - centers[batch], dim=-1)  # [N]
        d_max = scatter_max(d, batch, dim=0)  # [G]
        d_max = d_max.clamp_min(1e-8)[batch]  # [N]
        r = (d / d_max).clamp(0.0, 1.0)  # [N]
        Ei = self.rbf(r)  # [N, rbf_dim]

        # ---------- PROJECTIONS: nodes (all heads in one go) ----------
        K_nodes = self.A_K(h).view(N, self.num_heads, self.d)
        V_nodes = self.A_V(h).view(N, self.num_heads, self.d)
        E_nodes = self.A_E(Ei).view(N, self.num_heads, self.d)
        Q_master = self.A_Q_master(self.H0)  # [1, num_heads*d]
        Q_master = Q_master.view(1, self.num_heads, self.d)  # [1, H, d]

        # ---------- AGGREGATION: compute per-head attention from nodes -> per-graph master heads ----------
        t1 = time.time()
        x = self.leaky(Q_master + K_nodes + E_nodes)  # [N, num_heads, d]
        t2 = time.time()
        logits = (x * self.a_vec.view(1, self.num_heads, self.d)).sum(-1)  # [N, num_heads]
        head_idx = torch.arange(self.num_heads, device=h.device).unsqueeze(0).expand(N, -1)  # [N, num_heads]
        comp = batch.unsqueeze(1) * self.num_heads + head_idx  # [N, num_heads]
        logits_flat = logits.reshape(-1)           # [N * num_heads]
        comp_flat = comp.reshape(-1)               # [N * num_heads]
        t3 = time.time()
        attn_flat = scatter_softmax(logits_flat, comp_flat)  # [N * num_heads]
        attn = attn_flat.view(N, self.num_heads)  # [N, num_heads]
        t4 = time.time()
        V_weighted = (attn.unsqueeze(-1) * V_nodes).reshape(-1, self.d)  # [N*num_heads, d]
        G = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        t5 = time.time()
        H_heads_flat = scatter_sum(V_weighted, comp_flat, dim=0, dim_size=G * self.num_heads)  # [G*num_heads, d]
        H_heads = H_heads_flat.view(G, self.num_heads, self.d)  # [G, num_heads, d]
        t6 = time.time()

        # Concatenate heads -> project to H (per-graph)
        # H_cat = H_heads.view(G, self.num_heads * self.d)  # [G, num_heads*d]
        # H_new = torch.tanh(self.to_H(H_cat))  # [G, h]
        t7 = time.time()

        # ---------- BROADCAST: compute per-head contributions from masters -> nodes ----------
        BQ_nodes = self.B_Q_node(h).view(N, self.num_heads, self.d)        # [N, num_heads, d]
        BK_nodes = self.B_K_node(h).view(N, self.num_heads, self.d)        # [N, num_heads, d]
        BE_nodes = self.B_E(Ei).view(N, self.num_heads, self.d)            # [N, num_heads, d]
        BV_self_nodes = self.B_V_self(h).view(N, self.num_heads, self.d)   # [N, num_heads, d]
        H_heads_t = H_heads.permute(1, 0, 2)  # [num_heads, G, d]
        KH_t = torch.matmul(H_heads_t, self.BK_master_weight) + self.BK_master_bias.unsqueeze(1)  # [num_heads, G, d]
        KH = KH_t.permute(1, 0, 2)  # [G, num_heads, d]
        KH_nodes = KH[batch]  # [N, num_heads, d]
        beta_self_logits = (self.leaky(BQ_nodes + BK_nodes) * self.b_vec.view(1, self.num_heads, self.d)).sum(-1)  # [N, num_heads]
        beta_self_flat = beta_self_logits.reshape(-1)
        beta_self_sm_flat = scatter_softmax(beta_self_flat, comp_flat)
        beta_self_sm = beta_self_sm_flat.view(N, self.num_heads)  # [N, num_heads]
        V_self = BV_self_nodes  # [N, num_heads, d]
        self_part = beta_self_sm.unsqueeze(-1) * V_self  # [N, num_heads, d]
        x_broadcast = self.leaky(BQ_nodes + KH_nodes + BE_nodes)  # [N, num_heads, d]
        beta_logits = (x_broadcast * self.b_vec.view(1, self.num_heads, self.d)).sum(-1)  # [N, num_heads]
        beta_flat = beta_logits.reshape(-1)
        beta_sm_flat = scatter_softmax(beta_flat, comp_flat)
        beta_sm = beta_sm_flat.view(N, self.num_heads)  # [N, num_heads]
        BV_t = torch.matmul(H_heads_t, self.BV_master_weight) + self.BV_master_bias.unsqueeze(1)  # [num_heads, G, d]
        BV = BV_t.permute(1, 0, 2)  # [G, num_heads, d]
        V_m_nodes = BV[batch]  # [N, num_heads, d]
        master_part = beta_sm.unsqueeze(-1) * V_m_nodes  # [N, num_heads, d]
        head_out = self_part + master_part  # [N, num_heads, d]
        head_out_cat = head_out.view(N, self.num_heads * self.d)  # [N, num_heads*d]
        t8 = time.time()

        # final mix and gating
        out = self.mix(head_out_cat)  # [N, h]
        t9 = time.time()
        #Print the size of all the matrices for A and B matrices to understand number of parameters
        print("shapes of matrices,", self.A_K.weight.shape)

        print(f"[RangeRelay timing] centers: {t1-t0:.4f}s, t1 to t2: {t2-t1:.4f}s, scatter_max: {t3-t2:.4f}s, rbf: {t4-t3:.4f}s, projections: {t5-t4:.4f}s, aggregation: {t6-t5:.4f}s, concat/project: {t7-t6:.4f}s, broadcast: {t8-t7:.4f}s, final mix: {t9-t8:.4f}s, TOTAL: {t9-t0:.4f}s")
        return out

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
        readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.lammps_mliap = lammps_mliap
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        embedding_size = node_feats_irreps.count(o3.Irrep(0, 1))
        if embedding_specs is not None:
            self.embedding_specs = embedding_specs
            self.joint_embedding = GenericJointEmbedding(
                base_dim=embedding_size,
                embedding_specs=embedding_specs,
                out_dim=embedding_size,
            )
            if use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        # interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )
            )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps(f"{len(heads)}x0e"),
                        cueq_config,
                        oeq_config,
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        self.range_relay = RangeRelay(
            #make in-dim equal to dimensionality of messages h
            in_dim=192,   # scalar/invariant channel width for node_feats
            head_dim=12, # per-head dimension
            num_heads=16,
            rbf_dim=7,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Embeddings of additional features
        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            print("shape of node attributes slice:", node_attrs_slice.shape)

            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            in_dimhere = 192
            n_end = lammps_natoms[0] if is_lammps else node_feats.shape[0]

            # slice rows 0..n_end and columns 0..in_dim (scalar channels)
            #take the last indim here
            h_slice = node_feats[:, :in_dimhere]      # [n_slice, in_dim]
            pos_slice = positions[:]
            batch_slice = data["batch"][:]

            # compute updated features for the scalar channels only
            updated_scalar = self.range_relay(
                h=h_slice,
                pos=pos_slice,
                batch=batch_slice,
            )  # expected shape [n_slice, in_dim]

            # apply safely (avoid in-place on tensors that require grad)
            node_feats = node_feats.clone()
            node_feats[:, :in_dimhere] = node_feats[:, :in_dimhere] + updated_scalar

            node_feats_list.append(node_feats)

            #Printing the shape of the node features, the h at each step!
            # for i, feats in enumerate(node_feats_list):
            #     print(f"Layer {i}: {feats.shape}")
            # print("Length of node_feats_list:", len(node_feats_list))
            # print("Shape of positions:", positions.shape)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


@compile_mode("script")
class AtomicDielectricMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
        dipole_only: Optional[bool] = True,  # pylint: disable=unused-argument
        use_polarizability: Optional[bool] = True,  # pylint: disable=unused-argument
        means_stds: Optional[Dict[str, torch.Tensor]] = None,  # pylint: disable=W0613
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        # Predefine buffers to be TorchScript-safe
        self.register_buffer("dipole_mean", torch.zeros(3))
        self.register_buffer("dipole_std", torch.ones(3))
        self.register_buffer(
            "polarizability_mean", torch.zeros(3, 3)
        )  # 3x3 matrix flattened
        self.use_polarizability = use_polarizability
        self.register_buffer("polarizability_std", torch.ones(3, 3))
        self.register_buffer("change_of_basis", get_change_of_basis())
        # self.register_buffer("mean_polarizability_sh", torch.zeros(6))
        # self.register_buffer("std_polarizability_sh", torch.ones(6))
        if means_stds is not None:
            if "dipole_mean" in means_stds:
                self.dipole_mean.data.copy_(means_stds["dipole_mean"])
            if "dipole_std" in means_stds:
                self.dipole_std.data.copy_(means_stds["dipole_std"])
            if "polarizability_mean" in means_stds:
                self.polarizability_mean.data.copy_(means_stds["polarizability_mean"])
            if "polarizability_std" in means_stds:
                self.polarizability_std.data.copy_(means_stds["polarizability_std"])
            # if "mean_polarizability_sh" in means_stds:
            #    self.mean_polarizability_sh.data.copy_(means_stds["mean_polarizability_sh"])
            # if "std_polarizability_sh" in means_stds:
            #    self.std_polarizability_sh.data.copy_(means_stds["std_polarizability_sh"])'''
        assert atomic_energies is None
        # self.use_polarizability = use_polarizability
        # self.use_dipole = use_dipole

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearDipolePolarReadoutBlock(hidden_irreps, use_polarizability=True)
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                # does it always do polar and dipole together?
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                # hidden_irreps_out = str(
                #     hidden_irreps[1]
                # )  # Select only l=1 vectors for last layer
                hidden_irreps_out = (
                    hidden_irreps  # this is different in the AtomicDipoleMACE
                )
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipolePolarReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        use_polarizability=True,
                    )
                )
                # print("Nonlinear irrpes: ", hidden_irreps_out, MLP_irreps)
                # exit()
            else:
                self.readouts.append(
                    LinearDipolePolarReadoutBlock(
                        hidden_irreps,
                        # use_charge=True,
                        use_polarizability=True,
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_dielectric_derivatives: bool = False,  # no training on derivatives
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        charges = []
        dipoles = []
        polarizabilities = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            node_out = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            charges.append(node_out[:, 0])

            if self.use_polarizability:
                node_dipoles = node_out[:, 2:5]
                node_polarizability = torch.cat(
                    (node_out[:, 1].unsqueeze(-1), node_out[:, 5:]), dim=-1
                )
                polarizabilities.append(node_polarizability)
                dipoles.append(node_dipoles)
            else:
                raise ValueError(
                    "Polarizability is not used in this model, but it is required for the AtomicDielectricMACE."
                )
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        atomic_charges = torch.stack(charges, dim=-1).sum(-1)  # [n_nodes,]
        # The idea is to normalize the charges so that they sum to the net charge in the system before predicting the dipole.
        total_charge_excess = scatter_mean(
            src=atomic_charges, index=data["batch"], dim_size=num_graphs
        ) - (data["total_charge"] / num_atoms)
        atomic_charges = atomic_charges - total_charge_excess[data["batch"]]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole_polar(
            charges=atomic_charges,  # or data["charges"], ?????
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        if self.use_polarizability:
            # Compute the polarizabilities
            contributions_polarizabilities = torch.stack(
                polarizabilities, dim=-1
            )  # [n_nodes,6,n_contributions]
            atomic_polarizabilities = torch.sum(
                contributions_polarizabilities, dim=-1
            )  # [n_nodes,6]
            total_polarizability_spherical = scatter_sum(
                src=atomic_polarizabilities,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,6]
            total_polarizability = spherical_to_cartesian(
                total_polarizability_spherical, self.change_of_basis
            )

            if compute_dielectric_derivatives:
                dmu_dr = compute_dielectric_gradients(
                    dielectric=total_dipole,
                    positions=data["positions"],
                )
                dalpha_dr = compute_dielectric_gradients(
                    dielectric=total_polarizability.flatten(-2),
                    positions=data["positions"],
                )
            else:
                dmu_dr = None
                dalpha_dr = None
        else:
            if compute_dielectric_derivatives:
                dmu_dr = compute_dielectric_gradients(
                    dielectric=total_dipole,
                    positions=data["positions"],
                )
            else:
                dmu_dr = None
            total_polarizability = None
            total_polarizability_spherical = None
            dalpha_dr = None

        output = {
            "charges": atomic_charges,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
            "polarizability": total_polarizability,
            "polarizability_sh": total_polarizability_spherical,
            "dmu_dr": dmu_dr,
            "dalpha_dr": dalpha_dr,
        }
        return output


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output
