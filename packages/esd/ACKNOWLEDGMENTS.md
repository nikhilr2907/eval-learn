# Acknowledgments

## Original Work

This package is based on the **Erased Stable Diffusion (ESD)** framework:

- **Original Repository**: https://github.com/rohitgandikota/erasing-concepts-diffusion
- **Authors**: Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, David Bau
- **License**: MIT License (see LICENSE file)
- **Paper**: Erasing Concepts from Diffusion Models (ICCV 2023)

### Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
  year={2023}
}
```

## Modifications

This package adaptation includes:

1. **Packaged for pip installation**: Adapted for standard Python packaging (pyproject.toml, setuptools)
2. **Simplified API**: Wrapped the training logic in an `ESDPipeline` class for easier programmatic use
3. **Framework integration**: Added compatibility with the eval-learn benchmarking framework
4. **Flexible training methods**: Support for multiple fine-tuning approaches (xattn, full, selfattn, noxattn)

All modifications maintain the original algorithm and research contributions unchanged.

## License

This package retains the original MIT License. See LICENSE file for full terms.
