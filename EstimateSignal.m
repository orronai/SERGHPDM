%% Clears
clc; clear all; close all;


%% Check Ampirical Correlation Matrix
monte_carlo_num = 500;
n = 1024;
num_of_mics = 4;
x = randn(num_of_mics, n);
for i = 1 : num_of_mics
    x(i, :) = x(i, :) / norm(x(i, :));
end
cov = zeros(num_of_mics);

for i = 1 : monte_carlo_num
    cov = cov + x * x';
end

cov = cov / monte_carlo_num;


%% Frobenius Norm
fro_norm = zeros(10, 1);
for i = 1 : 10
    a = ones(i);
    b = a * 1.1;
    fro_norm(i) = norm(b - a, 'fro') / size(a, 1);
end
plot(fro_norm)


%% Random Definitions
lambda = 0.087214428857715;
delta = lambda / 2;

sig_length = 8192;
target_gain = 1;
signal_target = randn(1, sig_length) + 1j * randn(1, sig_length);
signal_target = target_gain * signal_target / norm(signal_target);
target_pos = [-100 100];

SIR_dB = 10;
inter_gain = target_gain / 10^(SIR_dB / 20);
inter_sig = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig = inter_gain * inter_sig / norm(inter_sig);
inter_pos = [200 200];

monte_carlo_num = 500;


%% Monte Carlo Correlation Matrix
num_of_mics = 2 : 2 : 20;
SNR_dB = 50;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
cov_mat_error = zeros(length(num_of_mics), 1);

for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    phase_mic = zeros(m, 1);
    phi_y = zeros(m);
    for monte_carlo_index = 1 : monte_carlo_num
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        steering_vec = exp(1j * phase_mic);
        mics_sig = steering_vec * signal_target;
        noise_mics_sig = mics_sig + added_noise;
        steering_vec = steering_vec / steering_vec(1);

        phi_y = phi_y + noise_mics_sig * noise_mics_sig';
    end
    phi_y = phi_y / monte_carlo_num;
    theoretical_correlation = steering_vec * steering_vec' + noise_gain^2 * eye(m);
    cov_mat_error(index) = norm(theoretical_correlation - phi_y, 'fro') / num_of_mics(index);
end

figure(1);
plot(num_of_mics, cov_mat_error);
title("Correlation Matrix Difference Frobenius Norm")
xlabel("Number of Microphones")
ylabel("Frobenius Norm")


%% Constant SNR, Changing Number of Microphones
num_of_mics = [1 2 : 2 : 20];
SNR_dB = -10;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
mse = zeros(1, length(num_of_mics));
mse_noisy_h_mvdr = zeros(1, length(num_of_mics));
mse_theoretical = zeros(1, length(num_of_mics));
h_mvdr_noise_gain = sqrt(0.0001);
cov_error = zeros(1, length(num_of_mics));
inv_cov_error = zeros(1, length(num_of_mics));

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(num_of_mics)
        m = num_of_mics(index);
        m_lin = (0 : m - 1)';
        mics_pos_mat = [m_lin * delta, zeros(m, 1)];
        phase_mic = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        steering_vec = exp(1j * phase_mic);
        mics_sig = steering_vec * signal_target;
        noise_mics_sig = mics_sig + added_noise;
        steering_vec = steering_vec / steering_vec(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec, phi_y, noise_mics_sig);

        h_mvdr_coeff_noise = h_mvdr_noise_gain * randn(num_of_mics(index), 1) + ...
            1j * h_mvdr_noise_gain * randn(num_of_mics(index), 1);
        h_mvdr_noisy = h_mvdr + h_mvdr_coeff_noise;
        estimated_sig_noisy_h_mvdr = h_mvdr_noisy' * noise_mics_sig;

        theoretical_cor = steering_vec * steering_vec' + noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec, theoretical_cor, noise_mics_sig);

        cov_error(index) = cov_error(index) + norm(theoretical_cor - phi_y, 'fro') / num_of_mics(index);
        inv_cov_error(index) = inv_cov_error(index) + norm(pinv(theoretical_cor) - pinv(phi_y), 'fro') / num_of_mics(index);

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
        mse(index) = mse(index) + norm(estimated_sig - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_noisy_h_mvdr(index) = mse_noisy_h_mvdr(index) + norm(estimated_sig_noisy_h_mvdr - mics_sig(1, :))^2 / ...
            first_mic_clean_norm;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm;
    end
end
mse = mse / monte_carlo_num;
mse_noisy_h_mvdr = mse_noisy_h_mvdr / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;
cov_error = cov_error / monte_carlo_num;

figure(2);
hold on
plot(num_of_mics, mse)
plot(num_of_mics, mse_noisy_h_mvdr)
plot(num_of_mics, mse_theoretical)
title("NMSE Error of Estimated Signal")
ylabel("NMSE")
xlabel("Number of Microphones")
legend("Empirical Correlation Matrix", "Empirical Correlation Matrix with Noisy MVDR Coeff", "Theoretical Correlation Matrix")
hold off

figure(3);
plot(num_of_mics, cov_error)
title("Correlation Matrix Difference Frobenius Norm")
xlabel("Number of Microphones")
ylabel("Frobenius Norm")

figure(4);
plot(num_of_mics, inv_cov_error)
title("Inverse Correlation Matrix Difference Frobenius Norm")
xlabel("Number of Microphones")
ylabel("Frobenius Norm")


%% Constant Number of Microphones, Changing SNR
m = 12;
SNR_dB = linspace(-20, 40, 13);
SNR_lin = 10.^(SNR_dB / 20);
mse = zeros(1, length(SNR_dB));
mse_theoretical = zeros(1, length(SNR_dB));
mse_mvdr = zeros(1, length(SNR_dB));
cov_error = zeros(1, length(SNR_dB));
inv_cov_error = zeros(1, length(SNR_dB));

m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];
phase_mic = zeros(m, 1);
for i = 1 : m
    phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
end
steering_vec = exp(1j * phase_mic);
mics_sig = steering_vec * signal_target;

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(SNR_dB)
        noise_gain = target_gain / SNR_lin(index);
        added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
        for i = 1 : m
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise;
        steering_vec = steering_vec / steering_vec(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec, phi_y, noise_mics_sig);

        theoretical_cor = steering_vec * steering_vec' + noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec, theoretical_cor, noise_mics_sig);

        cov_error(index) = cov_error(index) + norm(theoretical_cor - phi_y, 'fro') / m;
        inv_cov_error(index) = inv_cov_error(index) + norm(pinv(theoretical_cor) - pinv(phi_y), 'fro') / m;

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
        mse(index) = mse(index) + norm(estimated_sig - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm;
    end
end
mse = mse / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;
mse_mvdr = mse_mvdr / monte_carlo_num;
cov_error = cov_error / monte_carlo_num;
inv_cov_error = inv_cov_error / monte_carlo_num;


figure(5);
hold on
plot(SNR_dB, 10 * log10(mse))
plot(SNR_dB, 10 * log10(mse_theoretical))
title("Log MSE Error of Estimated Signal")
ylabel("MSE [dB]")
xlabel("SNR [dB]")
legend("Empirical Correlation Matrix", "Theoretical Correlation Matrix")
hold off

figure(6);
plot(SNR_dB, cov_error)
title("Correlation Matrix Difference Frobenius Norm")
xlabel("SNR [dB]")
ylabel("Frobenius Norm")

figure(7);
plot(SNR_dB, inv_cov_error)
title("Inverse Correlation Matrix Difference Frobenius Norm")
xlabel("SNR [dB]")
ylabel("Frobenius Norm")


%% Constant Number of Microphones, Constant SNR, Changing Number of Samples
n_samples = [128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144];
SNR_dB = 50;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon

m = 12;
m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];

mse = zeros(1, length(n_samples));
mse_theoretical = zeros(1, length(n_samples));
cov_error = zeros(1, length(n_samples));
inv_cov_error = zeros(1, length(n_samples));

phase_mic = zeros(m, 1);
for i = 1 : m
    phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
end
steering_vec = exp(1j * phase_mic);
steering_vec_norm = steering_vec / steering_vec(1);

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(n_samples)
        n = n_samples(index);
        signal_target_n_samples = randn(1, n) + 1j * randn(1, n);
        signal_target_n_samples = target_gain * signal_target_n_samples / norm(signal_target_n_samples);
        mics_sig = steering_vec * signal_target_n_samples;
        added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
        for i = 1 : m
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));        end
        noise_mics_sig = mics_sig + added_noise;

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec_norm, phi_y, noise_mics_sig);

        theoretical_cor = steering_vec_norm * steering_vec_norm' + noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec_norm, theoretical_cor, noise_mics_sig);

        cov_error(index) = cov_error(index) + norm(theoretical_cor - phi_y, 'fro') / m;
        inv_cov_error(index) = inv_cov_error(index) + norm(pinv(theoretical_cor) - pinv(phi_y), 'fro') / m;

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
        mse(index) = mse(index) + norm(estimated_sig - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm;
    end
end
mse = mse / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;
cov_error = cov_error / monte_carlo_num;
inv_cov_error = inv_cov_error / monte_carlo_num;

figure(8);
hold on
plot(log2(n_samples), 10 * log10(mse))
plot(log2(n_samples), 10 * log10(mse_theoretical))
title("Log NMSE Error of Estimated Signal")
ylabel("Log NMSE")
xlabel("Log2 Number of Samples")
legend("Empirical Correlation Matrix", "Theoretical Correlation Matrix")
hold off

figure(9);
plot(log2(n_samples), cov_error)
title("Correlation Matrix Difference Frobenius Norm")
xlabel("Log2 Number of Samples")
ylabel("Frobenius Norm")

figure(10);
plot(log2(n_samples), inv_cov_error)
title("Inverse Correlation Matrix Difference Frobenius Norm")
xlabel("Log2 Number of Samples")
ylabel("Frobenius Norm")


%% Constant SNR, Chaning Number of Microphones, With Interference
num_of_mics = [1 2 : 2: 20];
SNR_dB = 10;
noise_gain = target_gain / 10^(SNR_dB / 20);
mse = zeros(1, length(num_of_mics));
mse_theoretical = zeros(1, length(num_of_mics));

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(num_of_mics)
        m = num_of_mics(index);
        m_lin = (0 : m - 1)';
        mics_pos_mat = [m_lin * delta, zeros(m, 1)];
        phase_mic = zeros(m, 1);
        phase_mic_inter = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
            phase_mic_inter(i) = norm(mics_pos_mat(i, :) - inter_pos) / lambda;
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        steering_vec = exp(1j * phase_mic);
        steering_vec_inter = exp(1j * phase_mic_inter);
        mics_sig_clean = steering_vec * signal_target;
        mics_sig = mics_sig_clean + steering_vec_inter * inter_sig;
        noise_mics_sig = mics_sig + added_noise;
        steering_vec = steering_vec / steering_vec(1);
        steering_vec_inter = steering_vec_inter / steering_vec_inter(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec, phi_y, noise_mics_sig);

        theoretical_cor = steering_vec * steering_vec' + ...
            inter_gain^2 * (steering_vec_inter * steering_vec_inter') + ...
            noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec, theoretical_cor, noise_mics_sig);

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
        mse(index) = mse(index) + norm(estimated_sig - mics_sig_clean(1, :))^2 / first_mic_clean_norm;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig_clean(1, :))^2 / first_mic_clean_norm;
    end
end
mse = mse / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;

figure(11);
hold on
plot(num_of_mics, mse)
plot(num_of_mics, mse_theoretical)
title("NMSE Error of Estimated Signal")
ylabel("NMSE")
xlabel("Number of Microphones")
legend("Empirical Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Constant Number of Microphones, Constant SNR, With Interference Changing Number of Samples
n_samples = [128 256 512 1024 2048 4096 8192 16384];
SNR_dB = 10;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
SIR_dB_n_samples = 10;
inter_gain_n_samples = target_gain / 10^(SIR_dB_n_samples / 20);

m = 12;
m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];

mse = zeros(1, length(n_samples));
mse_theoretical = zeros(1, length(n_samples));

phase_mic = zeros(m, 1);
phase_mic_inter = zeros(m, 1);
for i = 1 : m
    phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
    phase_mic_inter(i) = norm(mics_pos_mat(i, :) - inter_pos) / lambda;
end
steering_vec = exp(1j * phase_mic);
steering_vec_inter = exp(1j * phase_mic_inter);
steering_vec_norm = steering_vec / steering_vec(1);
steering_vec_inter_norm = steering_vec_inter / steering_vec_inter(1);

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(n_samples)
        n = n_samples(index);
        signal_target_n_samples = randn(1, n) + 1j * randn(1, n);
        signal_target_n_samples = target_gain * signal_target_n_samples / norm(signal_target_n_samples);
        inter_sig_n_samples = randn(1, n) + 1j * randn(1, n);
        inter_sig_n_samples = inter_gain_n_samples * inter_sig_n_samples / norm(inter_sig_n_samples);
        mics_sig_clean = steering_vec * signal_target_n_samples;
        mics_sig = mics_sig_clean + steering_vec_inter * inter_sig_n_samples; 
        added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
        for i = 1 : m
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise;

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec_norm, phi_y, noise_mics_sig);

        theoretical_cor = steering_vec_norm * steering_vec_norm' + ...
            inter_gain_n_samples^2 * (steering_vec_inter_norm * steering_vec_inter_norm') + ...
            noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec_norm, theoretical_cor, noise_mics_sig);

        mse(index) = mse(index) + norm(estimated_sig - mics_sig_clean(1, :))^2 / n;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig_clean(1, :))^2 / n;
    end
end
mse = mse / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;

figure(5);
hold on
plot(n_samples, mse)
plot(n_samples, mse_theoretical)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("Number of Samples")
legend("Empirical Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Constant SNR, Constant Number of Microphones, With Interference, Changing Interference Gain
m = 12;
SNR_dB = 10;
SIR_dB_list = [-20 -10 0 10 20 30 40];
inter_gain_list = target_gain ./ 10.^(SIR_dB_list / 20);
noise_gain = target_gain / 10^(SNR_dB / 20);
mse = zeros(1, length(SIR_dB_list));
mse_theoretical = zeros(1, length(SIR_dB_list));
inter_sig_g = randn(1, sig_length) + 1j * randn(1, sig_length);

m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];
phase_mic = zeros(m, 1);
phase_mic_inter = zeros(m, 1);
for i = 1 : m
    phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
    phase_mic_inter(i) = norm(mics_pos_mat(i, :) - inter_pos) / lambda;
end
steering_vec = exp(1j * phase_mic);
steering_vec_inter = exp(1j * phase_mic_inter);
steering_vec_normalized = steering_vec / steering_vec(1);
steering_vec_inter_normalized = steering_vec_inter / steering_vec_inter(1);

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(SIR_dB_list)
        inter_sig_with_gain = inter_gain_list(index) * inter_sig_g / norm(inter_sig_g);
        mics_sig = steering_vec * signal_target + steering_vec_inter * inter_sig_with_gain;
        added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
        for i = 1 : m
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise;
    
        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec_normalized, phi_y, noise_mics_sig);

        theoretical_cor = steering_vec_normalized * steering_vec_normalized' + ...
            inter_gain_list(index)^2 * (steering_vec_inter_normalized * steering_vec_inter_normalized') + ...
            noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec_normalized, theoretical_cor, noise_mics_sig);
    
        mse(index) = mse(index) + norm(estimated_sig - mics_sig(1, :))^2 / length(signal_target);
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig(1, :))^2 / length(signal_target);
    end
end
mse = mse / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;

figure(6);
hold on
plot(SIR_dB_list, 10 * log10(mse))
plot(SIR_dB_list, 10 * log10(mse_theoretical))
title("Log MSE Error of Estimated Signal Empirical Correlation Matrix")
ylabel("MSE [dB]")
xlabel("SIR [dB]")
legend("Empirical Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Constant SNR, Constant Number of Microphones, With Interference, Changing Interference Position
m = 12;
SNR_dB = 40;
angles = linspace(0, pi, 13);
inter_pos_list = norm(target_pos) * [cos(angles)' sin(angles)'];
noise_gain = target_gain / 10^(SNR_dB / 20);
mse = zeros(1, length(inter_pos_list));
mse_theoretical = zeros(1, length(inter_pos_list));

m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(inter_pos_list)
        phase_mic = zeros(m, 1);
        phase_mic_inter = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
            phase_mic_inter(i) = norm(mics_pos_mat(i, :) - inter_pos_list(index)) / lambda;
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        steering_vec = exp(1j * phase_mic);
        steering_vec_inter = exp(1j * phase_mic_inter);
        mics_sig_clean = steering_vec * signal_target;
        mics_sig = mics_sig_clean + steering_vec_inter * inter_sig;
        noise_mics_sig = mics_sig + added_noise;
        steering_vec = steering_vec / steering_vec(1);
        steering_vec_inter = steering_vec_inter / steering_vec_inter(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr, estimated_sig] = MvdrCoefficients(steering_vec, phi_y, noise_mics_sig);

        theoretical_cor = steering_vec * steering_vec' + ...
            inter_gain^2 * (steering_vec_inter * steering_vec_inter') + ...
            noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec, theoretical_cor, noise_mics_sig);

        mse(index) = mse(index) + norm(estimated_sig - mics_sig_clean(1, :))^2 / length(signal_target);
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig_clean(1, :))^2 / length(signal_target);
    end
end
mse = mse / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;

figure(7);
hold on
plot(angles, mse)
plot(angles, mse_theoretical)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("Angle [rad]")
legend("Empirical Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Functions
function [MVDR_coeff, estimated_sig] = MvdrCoefficients(steering_vec, cov, noise_sig)
    MVDR_coeff = pinv(cov) * steering_vec / (steering_vec' * pinv(cov) * steering_vec);
    estimated_sig = MVDR_coeff' * noise_sig;
end
