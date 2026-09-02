import ase
import pytest
import torch

from mace.modules.radial import AgnesiTransform, SoftTransform, ZBLBasis


@pytest.fixture
def zbl_basis():
    return ZBLBasis(p=6, trainable=False)


def test_zbl_basis_initialization(zbl_basis):
    assert zbl_basis.p == torch.tensor(6.0)
    assert torch.allclose(zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))

    assert zbl_basis.a_exp == torch.tensor(0.300)
    assert zbl_basis.a_prefactor == torch.tensor(0.4543)
    assert not zbl_basis.a_exp.requires_grad
    assert not zbl_basis.a_prefactor.requires_grad


def test_trainable_zbl_basis_initialization(zbl_basis):
    zbl_basis = ZBLBasis(p=6, trainable=True)
    assert zbl_basis.p == torch.tensor(6.0)
    assert torch.allclose(zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))

    assert zbl_basis.a_exp == torch.tensor(0.300)
    assert zbl_basis.a_prefactor == torch.tensor(0.4543)
    assert zbl_basis.a_exp.requires_grad
    assert zbl_basis.a_prefactor.requires_grad


def test_forward(zbl_basis):
    x = torch.tensor([1.0, 1.0, 2.0]).unsqueeze(-1)  # [n_edges]
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]]
    )  # [n_nodes, n_node_features] - one_hot encoding of atomic numbers
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # [2, n_edges]
    atomic_numbers = torch.tensor([1, 6])  # [n_nodes]
    output = zbl_basis(x, node_attrs, edge_index, atomic_numbers)

    assert output.shape == torch.Size([node_attrs.shape[0]])
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor([0.0031, 0.0031], dtype=torch.get_default_dtype()),
        rtol=1e-2,
    )


@pytest.fixture
def agnesi():
    return AgnesiTransform(trainable=False)


def test_agnesi_transform_initialization(agnesi: AgnesiTransform):
    assert agnesi.q.item() == pytest.approx(0.9183, rel=1e-4)
    assert agnesi.p.item() == pytest.approx(4.5791, rel=1e-4)
    assert agnesi.a.item() == pytest.approx(1.0805, rel=1e-4)
    assert agnesi.prefactor.item() == pytest.approx(0.5, rel=1e-6)
    assert not agnesi.a.requires_grad
    assert not agnesi.q.requires_grad
    assert not agnesi.p.requires_grad
    assert not agnesi.prefactor.requires_grad


def test_trainable_agnesi_transform_initialization():
    agnesi = AgnesiTransform(trainable=True)

    assert agnesi.q.item() == pytest.approx(0.9183, rel=1e-4)
    assert agnesi.p.item() == pytest.approx(4.5791, rel=1e-4)
    assert agnesi.a.item() == pytest.approx(1.0805, rel=1e-4)
    assert agnesi.a.requires_grad
    assert agnesi.q.requires_grad
    assert agnesi.p.requires_grad


def test_agnesi_transform_forward():
    agnesi = AgnesiTransform()
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.get_default_dtype()).unsqueeze(-1)
    node_attrs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    atomic_numbers = torch.tensor([1, 6, 8])
    output = agnesi(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == x.shape
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor(
            [0.3646, 0.2175, 0.2089], dtype=torch.get_default_dtype()
        ).unsqueeze(-1),
        rtol=1e-2,
    )


def test_agnesi_transform_prefactor():
    agnesi_default = AgnesiTransform()
    agnesi_wide = AgnesiTransform(prefactor=1.0)
    assert agnesi_default.prefactor.item() == pytest.approx(0.5, rel=1e-6)
    assert agnesi_wide.prefactor.item() == pytest.approx(1.0, rel=1e-6)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.get_default_dtype()).unsqueeze(-1)
    node_attrs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    atomic_numbers = torch.tensor([1, 6, 8])
    out_default = agnesi_default(x, node_attrs, edge_index, atomic_numbers)
    out_wide = agnesi_wide(x, node_attrs, edge_index, atomic_numbers)
    assert not torch.allclose(out_default, out_wide)


def test_soft_transform_prefactor():
    soft_default = SoftTransform()
    soft_half = SoftTransform(prefactor=0.5)
    assert soft_default.prefactor.item() == pytest.approx(1.0, rel=1e-6)
    assert soft_half.prefactor.item() == pytest.approx(0.5, rel=1e-6)

    node_attrs = torch.tensor([[0, 1], [1, 0]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1], [1, 0]])
    atomic_numbers = torch.tensor([1, 6])

    r_0_default = soft_default.compute_r_0(node_attrs, edge_index, atomic_numbers)
    r_0_half = soft_half.compute_r_0(node_attrs, edge_index, atomic_numbers)
    r_cov_sum = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    assert torch.allclose(r_0_default, torch.full_like(r_0_default, float(r_cov_sum)))
    assert torch.allclose(r_0_half, 0.5 * r_0_default)

    x = torch.tensor([0.5, 1.5], dtype=torch.get_default_dtype()).unsqueeze(-1)
    out_default = soft_default(x, node_attrs, edge_index, atomic_numbers)
    out_half = soft_half(x, node_attrs, edge_index, atomic_numbers)
    assert not torch.allclose(out_default, out_half)


def test_soft_transform_forward():
    soft = SoftTransform()
    node_attrs = torch.tensor([[0, 1], [1, 0]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1], [1, 0]])
    atomic_numbers = torch.tensor([1, 6])
    x = torch.tensor([0.5, 1.5], dtype=torch.get_default_dtype()).unsqueeze(-1)
    output = soft(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == x.shape
    assert torch.is_tensor(output)
    # deep bond lengths are compressed towards p_0 = 0.75 * r_0
    r_0 = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    assert output[0].item() == pytest.approx(0.75 * r_0, abs=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
