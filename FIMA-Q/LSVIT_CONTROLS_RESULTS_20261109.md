# LS reconstruction controls, 2026-11-09

All three ViT-S W3A3 seed 3407 controls finished fresh 20,000-step reconstruction per block. They use the same floating-point model SHA256 (`a78205f2...e513463c`), the same 128/1024 calibration image and augmented-tensor hashes, the same 3-bit configuration, and the same streaming damped LS loss. The controls used the same source fingerprint (`e32b67b5a875...`).

| Control | Top-1 | Top-5 | Validation loss | Top-1 versus fixed |
| --- | ---: | ---: | ---: | ---: |
| Plain LS, no rotation | 64.404% | 87.100% | 1.441552 | -1.156 pp |
| Fixed Hadamard + LS | **65.560%** | 87.796% | 1.397692 | reference |
| Fixed Hadamard + B-MSE pre-calibration + LS | 65.314% | **87.914%** | **1.388182** | -0.246 pp |
| Historical fixed Hadamard + C-MSE pre-calibration + LS | 65.022% | 87.462% | 1.410583 | -0.538 pp, cross-run |

The paired fixed-minus-plain gain is +1.156 pp. B accepted changes in 7 of 12 blocks (1, 3, 4, 5, 7, 8, 10). Its local teacher KL improved where accepted, but final Top-1 fell by 0.246 pp. Its lower validation loss and higher Top-5 do not compensate for the Top-1 drop under the current objective. The local pre-calibration gate is therefore not a reliable predictor of final Top-1 after reconstruction.

The historical C run accepted 10 of 12 blocks and reached 65.022%. Its floating-point model and calibration hashes match, and the later control-entry diff did not change the LS numerical loss. Nevertheless, it has a different run/code fingerprint (`8f57260c6d8b...`) and was not rerun alongside these three controls. Therefore -0.538 pp is a cross-run reference, not a paired estimate of the learned rotation effect. A current-protocol C run is the shortest decisive check if that mechanism remains under consideration.

The best recorded score in the synchronized results is fixed Hadamard + LS at 65.560%. Its checkpoint path is recorded in `method_result.json`; the weight file itself was not synchronized locally. This combines known mechanisms and alone is an empirical baseline rather than a confirmed new algorithm. Historical fixed Hadamard + DPLR was 64.650%, but its rank and reconstruction estimator differ, so the +0.910 pp numerical gap cannot be attributed solely to replacing DPLR with LS. Do not claim statistical stability from one seed.

Evidence: `checkpoints/lsvit_controls/summary_seed3407_e32b67b5a875.json`, the three `20261109_1333*_lscontrol_*` `method_result.json` and `output.log` files, and `checkpoints/quant_result/20261104_193139_method_lsvit_4373e8b51261/method_result.json` for historical C.
