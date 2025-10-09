"""Tests for the correctness of methods in methods.py.

This module tests the EngressionNet architecture and engression_loss function
to ensure they work correctly and produce expected outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import numpy as np
from methods import get_device, EngressionNet, engression_loss


class TestGetDevice:
    """Test device selection logic."""
    
    def test_get_device_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_get_device_is_valid(self):
        device = get_device()
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_get_device_can_allocate_tensor(self):
        device = get_device()
        tensor = torch.zeros(1, device=device)
        assert tensor.device.type == device.type


class TestEngressionNetArchitecture:
    """Test EngressionNet structure and forward pass."""
    
    def test_network_initialization_default(self):
        model = EngressionNet()
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'network')
    
    def test_network_forward_pass_default(self):
        device = get_device()
        model = EngressionNet().to(device)
        x = torch.randn(10, 2, device=device)
        output = model(x)
        assert output.shape == (10, 2)
    
    def test_network_forward_pass_custom_dims(self):
        device = get_device()
        model = EngressionNet(input_dim=5, output_dim=3, hidden_dim=32).to(device)
        x = torch.randn(8, 5, device=device)
        output = model(x)
        assert output.shape == (8, 3)
    
    def test_network_with_dropout(self):
        device = get_device()
        model = EngressionNet(dropout=0.5).to(device)
        x = torch.randn(10, 2, device=device)
        model.train()
        output = model(x)
        assert output.shape == (10, 2)
    
    def test_network_with_batchnorm(self):
        device = get_device()
        model = EngressionNet(use_batchnorm=True, num_layers=2).to(device)
        x = torch.randn(10, 2, device=device)
        output = model(x)
        assert output.shape == (10, 2)
    
    def test_network_num_layers(self):
        device = get_device()
        model = EngressionNet(num_layers=5, hidden_dim=128).to(device)
        x = torch.randn(10, 2, device=device)
        output = model(x)
        assert output.shape == (10, 2)
    
    def test_network_gradient_flow(self):
        device = get_device()
        model = EngressionNet().to(device)
        x = torch.randn(10, 2, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestEngressionLoss:
    """Test engression loss computation."""
    
    def test_loss_returns_three_values(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        result = engression_loss(model, X_batch, m=5, device=device)
        assert len(result) == 3
    
    def test_loss_returns_scalars(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
        assert loss.dim() == 0
        assert term1.dim() == 0
        assert term2.dim() == 0
    
    def test_loss_is_positive(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
        assert loss.item() >= 0
        assert term1.item() >= 0
        assert term2.item() >= 0
    
    def test_loss_with_different_batch_sizes(self):
        device = get_device()
        model = EngressionNet().to(device)
        
        for batch_size in [1, 5, 10, 32]:
            X_batch = torch.randn(batch_size, 2, device=device)
            loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
            assert loss.item() >= 0
    
    def test_loss_with_different_m_values(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        
        for m in [3, 5, 10, 20]:  # m must be >= 2 to avoid division by zero
            loss, term1, term2 = engression_loss(model, X_batch, m=m, device=device)
            assert loss.item() >= 0
            assert not torch.isnan(loss)
    
    def test_loss_without_explicit_device(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=5)
        assert loss.item() >= 0
    
    def test_loss_term1_term2_relationship(self):
        """Test that loss combines term1 and term2 correctly."""
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
        
        # loss should be approximately term1 - 0.5*term2
        expected_loss = term1 - 0.5 * term2
        assert torch.allclose(loss, expected_loss, atol=1e-6)


class TestEngressionLossMathematicalProperties:
    """Test mathematical properties of the loss function."""
    
    def test_term1_dimension(self):
        """Test that term1 has correct dimensions."""
        device = get_device()
        model = EngressionNet().to(device)
        batch_size = 10
        X_batch = torch.randn(batch_size, 2, device=device)
        _, term1, _ = engression_loss(model, X_batch, m=5, device=device)
        
        # term1 should be a scalar (averaged over batch)
        assert term1.dim() == 0
    
    def test_term2_dimension(self):
        """Test that term2 has correct dimensions."""
        device = get_device()
        model = EngressionNet().to(device)
        batch_size = 10
        X_batch = torch.randn(batch_size, 2, device=device)
        _, _, term2 = engression_loss(model, X_batch, m=5, device=device)
        
        # term2 should be a scalar (averaged over batch)
        assert term2.dim() == 0
    
    def test_loss_symmetry_in_m(self):
        """Test that loss computation is consistent."""
        device = get_device()
        torch.manual_seed(42)
        model = EngressionNet().to(device)
        X_batch = torch.randn(5, 2, device=device)
        
        torch.manual_seed(123)
        loss1, _, _ = engression_loss(model, X_batch, m=5, device=device)
        
        torch.manual_seed(123)
        loss2, _, _ = engression_loss(model, X_batch, m=5, device=device)
        
        assert torch.allclose(loss1, loss2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_batch(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(1, 2, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
        assert loss.item() >= 0
    
    def test_single_epsilon_sample(self):
        device = get_device()
        model = EngressionNet().to(device)
        X_batch = torch.randn(10, 2, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=2, device=device)
        # With m=2, term2 should be small but well-defined
        assert not torch.isnan(term2)
        assert term2.item() >= 0
    
    def test_model_evaluation_mode(self):
        device = get_device()
        model = EngressionNet(dropout=0.5, use_batchnorm=True).to(device)
        X_batch = torch.randn(10, 2, device=device)
        
        model.eval()
        with torch.no_grad():
            loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
        
        assert loss.item() >= 0
    
    def test_high_dimensional_input(self):
        device = get_device()
        model = EngressionNet(input_dim=100, output_dim=50).to(device)
        X_batch = torch.randn(10, 50, device=device)
        loss, term1, term2 = engression_loss(model, X_batch, m=5, device=device)
        assert loss.item() >= 0