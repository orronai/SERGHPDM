%% Clears
clc; clear all; close all;


%% Random Definitions
lambda = 0.087214428857715;
delta = lambda / 2;

sig_length = 1024;
target_gain = 1;
signal_target = randn(1, sig_length) + 1j * randn(1, sig_length);
signal_target = target_gain * signal_target / norm(signal_target);
target_pos = [-1 1];

inter_gain = 1e-2;
inter_sig = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig = 0 * inter_gain * inter_sig / norm(inter_sig);
inter_pos = [2 2];

%% Constant SNR, Changing Number of Microphones
num_of_mics = [4 8 12 24 36 48 72 96 128 256];
SNR_dB = 10;
noise_gain = target_gain / 10^(SNR_dB / 10);
mse = zeros(1, length(num_of_mics));

for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    phase_mic = zeros(m, 1);
    for i = 1 : m
        phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
    end
    steering_vec = exp(1j * phase_mic);
    mics_sig = steering_vec * signal_target;
    added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
    noise_mics_sig = mics_sig + noise_gain * (added_noise / norm(added_noise));
    steering_vec = steering_vec / steering_vec(1);

%     phi_y = noise_mics_sig * noise_mics_sig';
    phi_y = mean(abs(signal_target.^2)) * (steering_vec * steering_vec') + ...
        added_noise * added_noise';
    
    h_mvdr = pinv(phi_y) * steering_vec / (steering_vec' * pinv(phi_y) * steering_vec);

    estimated_sig = h_mvdr' * noise_mics_sig;
    
    mse(index) = norm(estimated_sig - mics_sig(1, :))^2 / length(signal_target);
end

figure(2);
plot(num_of_mics, mse)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("Number of Microphones")


%% Constant Number of Microphones, Changing SNR
m = 12;
SNR_dB = linspace(-30, 50, 9);
SNR_lin = 10.^(SNR_dB / 10);
mse = zeros(1, length(SNR_dB));

m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];
phase_mic = zeros(m, 1);
for i = 1 : m
    phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
end
steering_vec = exp(1j * phase_mic);
mics_sig = steering_vec * signal_target;

for index = 1 : length(SNR_dB)
    noise_gain = target_gain / SNR_lin(index);
    added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
    noise_mics_sig = mics_sig + noise_gain * (added_noise / norm(added_noise));
    steering_vec = steering_vec / steering_vec(1);

%     phi_y = noise_mics_sig * noise_mics_sig';
    phi_y = mean(abs(signal_target.^2)) * (steering_vec * steering_vec') + ...
        added_noise * added_noise';
    
    h_mvdr = pinv(phi_y) * steering_vec / (steering_vec' * pinv(phi_y) * steering_vec);

    estimated_sig = h_mvdr' * noise_mics_sig;
    
    mse(index) = norm(estimated_sig - mics_sig(1, :))^2 / length(signal_target);
end

figure(2);
plot(SNR_dB, mse)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("SNR [dB]")


%% Constant SNR, Chaning Number of Microphones, With Interference
num_of_mics = [4 8 12 24 36 48 72 96 128 256];
SNR_dB = 10;
noise_gain = target_gain / 10^(SNR_dB / 10);
mse = zeros(1, length(num_of_mics));

for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
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
    mics_sig = steering_vec * signal_target + steering_vec_inter * inter_sig;
    added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
    noise_mics_sig = mics_sig + noise_gain * (added_noise / norm(added_noise));
    steering_vec = steering_vec / steering_vec(1);

    phi_y = noise_mics_sig * noise_mics_sig';
%     phi_y = mean(abs(signal_target.^2)) * (steering_vec * steering_vec') + ...
%         mean(abs(inter_sig.^2)) * (steering_vec_inter * steering_vec_inter') + ...
%         added_noise * added_noise';

    h_mvdr = pinv(phi_y) * steering_vec / (steering_vec' * pinv(phi_y) * steering_vec);

    estimated_sig = h_mvdr' * noise_mics_sig;

    mse(index) = norm(estimated_sig - mics_sig(1, :))^2 / length(signal_target);
end

figure(3);
plot(num_of_mics, mse)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("Number of Microphones")


%% Constant SNR, Constant Number of Microphones, With Interference, Changing Interference Gain
m = 12;
SNR_dB = 10;
SNR_inter_dB = [-30 -20 -10 0 10 20 30 40];
inter_gain_list = target_gain ./ 10.^(SNR_inter_dB / 10);
noise_gain = target_gain / 10^(SNR_dB / 10);
mse = zeros(1, length(SNR_inter_dB));
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

for index = 1 : length(SNR_inter_dB)
    inter_sig_with_gain = inter_gain_list(index) * inter_sig_g / norm(inter_sig_g);
    mics_sig = steering_vec * signal_target + steering_vec_inter * inter_sig_with_gain;
    added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
    noise_mics_sig = mics_sig + noise_gain * (added_noise / norm(added_noise));
    steering_vec = steering_vec / steering_vec(1);

%     phi_y = noise_mics_sig * noise_mics_sig';
    phi_y = mean(abs(signal_target.^2)) * (steering_vec * steering_vec') + ...
        mean(abs(inter_sig_with_gain.^2)) * (steering_vec_inter * steering_vec_inter') + ...
        added_noise * added_noise';

    h_mvdr = pinv(phi_y) * steering_vec / (steering_vec' * pinv(phi_y) * steering_vec);

    estimated_sig = h_mvdr' * noise_mics_sig;

    mse(index) = norm(estimated_sig - mics_sig(1, :))^2 / length(signal_target);
end

figure(4);
plot(SNR_inter_dB, mse)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("SNR Intereference [dB]")


%% Constant SNR, Constant Number of Microphones, With Interference, Changing Interference Position
m = 12;
SNR_dB = 20;
angles = linspace(0, pi);
inter_pos_list = norm(target_pos) * [cos(angles)' sin(angles)'];
noise_gain = target_gain / 10^(SNR_dB / 10);
mse = zeros(1, length(inter_pos_list));

m_lin = (0 : m - 1)';
mics_pos_mat = [m_lin * delta, zeros(m, 1)];

for index = 1 : length(inter_pos_list)
    phase_mic = zeros(m, 1);
    phase_mic_inter = zeros(m, 1);
    for i = 1 : m
        phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
        phase_mic_inter(i) = norm(mics_pos_mat(i, :) - inter_pos_list(index)) / lambda;
    end
    steering_vec = exp(1j * phase_mic);
    steering_vec_inter = exp(1j * phase_mic_inter);
    mics_sig = steering_vec * signal_target + steering_vec_inter * inter_sig;
    added_noise = randn(size(mics_sig)) + 1j * randn(size(mics_sig));
    noise_mics_sig = mics_sig + noise_gain * (added_noise / norm(added_noise));
    steering_vec = steering_vec / steering_vec(1);

%     phi_y = noise_mics_sig * noise_mics_sig';
    phi_y = mean(abs(signal_target.^2)) * (steering_vec * steering_vec') + ...
        mean(abs(inter_sig.^2)) * (steering_vec_inter * steering_vec_inter') + ...
        added_noise * added_noise';

    h_mvdr = pinv(phi_y) * steering_vec / (steering_vec' * pinv(phi_y) * steering_vec);

    estimated_sig = h_mvdr' * noise_mics_sig;

    mse(index) = norm(estimated_sig - mics_sig(1, :))^2 / length(signal_target);
end

figure(5);
plot(angles, mse)
title("MSE Error of Estimated Signal")
ylabel("MSE")
xlabel("Angle [rad]")

