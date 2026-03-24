# Acknowledgments

## Original Work

This package is based on the **MACE** (Mass Concept Erasure in Diffusion Models) framework:

- **Original Repository**: [MACE on GitHub](https://github.com/Shilin-LU/MACE)
- **Authors**: Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, Adams Wai-Kin Kong
- **Institution**: Nanyang Technological University (NTU) - NTUITIVE PTE LTD
- **License**: NTUITIVE Non-Commercial Dual License (see LICENSE file)
- **Paper**: Mass Concept Erasure in Diffusion Models (CVPR 2024)

### Citations

If you use this code in your research, please cite:

**Primary Paper:**

```bibtex
@inproceedings{lu2024mace,
  title={Mace: Mass concept erasure in diffusion models},
  author={Lu, Shilin and Wang, Zilan and Li, Leyang and Liu, Yanzhu and Kong, Adams Wai-Kin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6430--6440},
  year={2024}
}
```

**Related Work:**

```bibtex
@article{li2025set,
  title={Set you straight: Auto-steering denoising trajectories to sidestep unwanted concepts},
  author={Li, Leyang and Lu, Shilin and Ren, Yan and Kong, Adams Wai-Kin},
  journal={arXiv preprint arXiv:2504.12782},
  year={2025}
}
```

### Contact for Commercial Use

If you wish to obtain a commercial royalty-bearing license to this software, please contact:

**shilin002@e.ntu.edu.sg**

## License

⚠️ **IMPORTANT**: This software is licensed under the **NTUITIVE Non-Commercial License**.

**Non-Commercial Use Only**: This software may only be used for:
- Teaching
- Academic research
- Public demonstrations
- Personal experimentation

**Commercial use is prohibited** without explicit written permission from NTUITIVE.

See LICENSE file for full terms and conditions.

## Modifications

This package adaptation includes:

1. **Packaged for pip installation**: Adapted for standard Python packaging (pyproject.toml, setuptools)
2. **Framework integration**: Added compatibility with the eval-learn benchmarking framework
3. **API wrapper**: Created MACEPipeline class for streamlined usage
4. **Weight management**: Support for saving and loading trained weights

All modifications maintain the original algorithm and research contributions unchanged.

## Compliance

By using this software, you agree to comply with the NTUITIVE Non-Commercial License terms. For questions regarding license compliance or commercial use, contact shilin002@e.ntu.edu.sg.
