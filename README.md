|x-GAN               |Links                           |
|--------------------|--------------------------------|
| GAN                |https://arxiv.org/pdf/1406.2661 |
|wGAN                |https://arxiv.org/pdf/1701.07875|
|Least Squares GAN   |https://arxiv.org/pdf/1611.04076|





---

- Learning rate scheduler 정리   
> https://sanghyu.tistory.com/113

------------------------------------
I'll add a short todo tracking this recommendation task, then give concise simulation suggestions and brief setups.


Progress: todo recorded. Below are concise simulation ideas you can use as GAN inputs, with why and minimal setup notes.

Multipath Rayleigh taps (time-domain CSI)

Why: realistic small-scale fading; GAN can learn channel realizations for augmentation.
Setup: generate tapped-delay line with L taps, complex Gaussian taps (Rayleigh), random delays; sample impulse responses length N; split real/imag into two channels.
Rician / LOS with K-factor sweep

Why: covers mixture of LOS and scatter—useful for conditional GANs.
Setup: add deterministic LOS phasor + Rayleigh taps; condition on K-factor label.
Frequency-selective OFDM subcarrier CSI (antenna × subcarrier map)

Why: natural 2D “image” input (antennas vs subcarriers) for CNN-GANs.
Setup: simulate OFDM (Nfft, CP), apply channel frequency response per subcarrier; store complex H as 2-channel image.
Time-varying Doppler channel sequences

Why: captures temporal correlation (mobility); good for video-GANs or RNN-GANs.
Setup: Jakes/Clarke Doppler model over T time steps; produce sequence of CSI frames.
Raw IQ traces of received OFDM symbols (with impairments)

Why: end-to-end receiver tasks (denoising, adversarial training).
Setup: generate transmitted OFDM frames, apply channel + AWGN + CFO/phase-noise, store complex IQ sequence (split into real/imag).
Constellation-density images (2D histograms)

Why: compact representation for modulation patterns and impairments (nonlinearity, clipping).
Setup: collect received symbol scatter, rasterize into heatmap images for GAN training.
Pilot-based channel estimates (sparse measurements → super-resolution)

Why: train GAN to hallucinate full CSI from sparse pilots.
Setup: simulate pilot positions and noisy pilot estimates as input; target is full estimated CSI.
MIMO channel matrices (Tx × Rx × subcarriers)

Why: high-dimensional realistic data for MIMO receivers; conditional on array geometry.
Setup: simulate correlated MIMO taps (Kronecker model or physical ray tracing), format as multi-channel tensor.
Nonlinear RF impairments / PA distortion on OFDM (PAPR effects)

Why: useful for training GAN-based pre-distortion or robustness.
Setup: apply memoryless AM/AM & AM/PM model or Saleh model to OFDM time signal; GAN learns inversion/compensation.
BER/FER curves or performance maps (SNR × parameter grid)

Why: GAN can interpolate performance surfaces for fast system-level emulation.
Setup: simulate link performance across SNR, fading, modulation; store as small images/grids.
Practical tips

Represent complex-valued data by two real channels (real, imag) or magnitude+phase; normalize per-dataset.
Use conditional labels (SNR, Doppler, K-factor, #taps) for cGAN to control outputs.
Typical input sizes: CSI images (antennas × subcarriers) e.g., 4×64 or 8×128; IQ sequences length 1024–8192 for OFDM frames.
Data augmentation: randomize delay spreads, pilot patterns, SNR, and frequency offsets to improve GAN generalization.
Tools: implement sims with NumPy/SciPy (tapped-delay, AWGN), or use comms libs (e.g., commpy) and PyTorch for GANs.
If you want, I can:

implement one simulation generator (e.g., OFDM IQ + AWGN + CFO) as a Python script,
or patch your notebook to add a data-generation cell for a chosen simulation.