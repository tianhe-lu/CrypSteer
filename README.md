# CrypSteer

Code and data for Paper: [Many Repairs, One Direction: Steering Code Models Toward Safer Cryptographic API Use](https://arxiv.org/abs/2610.25133).

- [Data](data/README.md): dataset files, contents, and fields.
- [Experiments](scripts/README.md): dependencies, script order, and commands. Run commands from this repository root.
- [Evaluator](evaluator/README.md): security and structural-validity checks and toolchain setup.

## License

The code and datasets in this repository are released under the [MIT License](LICENSE). Third-party components retain their original licenses.

## Citation

If you use the code or datasets in your research, please cite our paper:

@misc{lu2026repairsdirectionsteeringcode,
      title={Many Repairs, One Direction: Steering Code Models Toward Safer Cryptographic API Use}, 
      author={Tianhe Lu and Sakuna Harinda Jayasundara and Eric Spero and Ewan Tempero and Robert Biddle and Giovanni Russello},
      year={2026},
      eprint={2610.25133},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2610.25133}, 
}

## Dataset setup

From the repository root, extract the dataset before running the scripts:

```bash
unzip crypsteer_data.zip
