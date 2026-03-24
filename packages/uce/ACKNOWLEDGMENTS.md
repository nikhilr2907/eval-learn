# Acknowledgments

## Original Work

This package is based on the **Unified Concept Editing (UCE)** framework:

- **Original Repository**: https://github.com/rohitgandikota/unified-concept-editing
- **Authors**: Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzynska, David Bau
- **License**: MIT License (see LICENSE file)
- **Paper**: Unified Concept Editing in Diffusion Models (CVPR 2024)

### Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{gandikota2023unified,
  title={Unified Concept Editing in Diffusion Models},
  author={Gandikota, Rohit and Orgad, Hadas and Belinkov, Yonatan and Materzynska, Joanna and Bau, David},
  journal={arXiv preprint arXiv:2308.14761},
  year={2023}
}
```

## Modifications

This package adaptation includes:

1. **Self-contained training script**: Bundled `trainscripts/uce_sd_erase.py` directly in the package to eliminate the need for cloning the external repository
2. **Simplified API**: Wrapped the training script in a `UCEWeightCreator` class for easier programmatic use
3. **Pre-trained weights**: Included bundled pre-trained weights (nudity, violence, dog) for common concepts
4. **Package structure**: Adapted for pip installation with standard Python packaging (pyproject.toml, setuptools)
5. **Integration**: Added compatibility with the eval-learn benchmarking framework

All modifications maintain the original algorithm and research contributions unchanged.

## License

This package retains the original MIT License. See LICENSE file for full terms.
