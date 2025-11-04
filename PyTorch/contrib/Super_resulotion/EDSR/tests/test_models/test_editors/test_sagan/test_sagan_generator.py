# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch
import torch_sdaa

from mmagic.models.editors.sagan import SNGANGenerator
from mmagic.registry import MODELS


class TestSNGANPROJGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((2, 128))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SNGANGenerator',
            noise_size=128,
            output_scale=32,
            num_classes=10,
            base_channels=32)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-sdaa due to limited RAM.')
    def test_sngan_proj_generator(self):

        # test default setting with builder
        g = MODELS.build(self.default_config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test return noise
        x = g(None, num_batches=2, return_noise=True)
        assert x['fake_img'].shape == (2, 3, 32, 32)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(self.noise, label=self.label, return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(torch.randn, num_batches=2, return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        # test different output_scale
        config = deepcopy(self.default_config)
        config['output_scale'] = 64
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 64, 64)

        # test num_classes == 0 and `use_cbn = True`
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        with pytest.raises(ValueError):
            g = MODELS.build(config)

        # test num_classes == 0 and `use_cbn = False`
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        config['use_cbn'] = False
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 64
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [16, 8, 4, 2]}
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> error (key not find)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {64: [16, 8, 4, 2]}
        with pytest.raises(KeyError):
            g = MODELS.build(config)

        # test different channels_cfg --> error (type not match)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = '1234'
        with pytest.raises(ValueError):
            g = MODELS.build(config)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test with_embedding_spectral_norm
        config = deepcopy(self.default_config)
        config['with_embedding_spectral_norm'] = True
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test norm_eps
        config = deepcopy(self.default_config)
        config['norm_eps'] = 1e-9
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test sn_eps
        config = deepcopy(self.default_config)
        config['sn_eps'] = 1e-12
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> Studio
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='studio')
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> SNGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> SAGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        g = MODELS.build(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            g = MODELS.build(config)

        # test pretrained --> raise error
        config = deepcopy(self.default_config)
        config['pretrained'] = 42
        with pytest.raises(TypeError):
            g = MODELS.build(config)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_sngan_proj_generator_cuda(self):

        # test default setting with builder
        g = MODELS.build(self.default_config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test return noise
        x = g(None, num_batches=2, return_noise=True)
        assert x['fake_img'].shape == (2, 3, 32, 32)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(self.noise.sdaa(), label=self.label.sdaa(), return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(torch.randn, num_batches=2, return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        # test different output_scale
        config = deepcopy(self.default_config)
        config['output_scale'] = 64
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 64, 64)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 64
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [16, 8, 4, 2]}
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test with_embedding_spectral_norm
        config = deepcopy(self.default_config)
        config['with_embedding_spectral_norm'] = True
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test norm_eps
        config = deepcopy(self.default_config)
        config['norm_eps'] = 1e-9
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test sn_eps
        config = deepcopy(self.default_config)
        config['sn_eps'] = 1e-12
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2).sdaa()
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> SNGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        g = MODELS.build(config).sdaa()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
