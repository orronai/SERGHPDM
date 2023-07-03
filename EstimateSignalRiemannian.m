%% Clears
clc; clear all; close all;


%% Random Definitions
lambda = 0.087214428857715;
delta = lambda / 2;
trans_dist = 30;
inter_1_dist = 30;
inter_2_dist = 30;

sig_length = 1024;
target_gain = 1;
signal_target = randn(1, sig_length) + 1j * randn(1, sig_length);
signal_target = target_gain * signal_target / norm(signal_target);
target_angle = 35.6;
target_pos = trans_dist * [cosd(target_angle) sind(target_angle)];

SIR_dB_1 = -15;
inter_gain_1 = target_gain / 10^(SIR_dB_1 / 20);
inter_sig_1 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_1 = inter_gain_1 * inter_sig_1 / norm(inter_sig_1);
inter_angle_1 = 61.3;
inter_pos_1 = inter_1_dist * [cosd(inter_angle_1) sind(inter_angle_1)];

SIR_dB_2 = -10;
inter_gain_2 = target_gain / 10^(SIR_dB_2 / 20);
inter_sig_2 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_2 = inter_gain_2 * inter_sig_2 / norm(inter_sig_2);
inter_angle_2 = 11.4;
inter_pos_2 = inter_2_dist * [cosd(inter_angle_2) sind(inter_angle_2)];

mask = [ones(1, sig_length / 2), zeros(1, sig_length / 2)];
inter_sig_1 = inter_sig_1 .* mask;
inter_sig_2 = inter_sig_2 .* (1 - mask);

monte_carlo_num = 500;


%% Constant SNR, Changing Number of Microphones, Without Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 0;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
log_nmse_E = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_R = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_theoretical = zeros(1, length(num_of_mics));


for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    distance_trans = zeros(m, 1);
    distance_inter_1 = zeros(m, 1);
    distance_inter_2 = zeros(m, 1);
    for i = 1 : m
        distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
        distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
        distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
    end
    phase_mic = distance_trans / lambda;
    phase_mic_inter_1 = distance_inter_1 / lambda;
    phase_mic_inter_2 = distance_inter_2 / lambda;
    steering_vec = exp(-1j * 2 * pi * phase_mic);
    steering_vec_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1);
    steering_vec_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2);
    mics_sig = steering_vec * signal_target;
    mics_inter_sig_1 = steering_vec_inter_1 * inter_sig_1;
    mics_inter_sig_2 = steering_vec_inter_2 * inter_sig_2;
    steering_vec = steering_vec / steering_vec(1);
    steering_vec_inter_1 = steering_vec_inter_1 / steering_vec_inter_1(1);
    steering_vec_inter_2 = steering_vec_inter_2 / steering_vec_inter_2(1);
    first_mic_clean_norm = norm(mics_sig(1, :))^2;

    for monte_carlo_index = 1 : monte_carlo_num
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise + mics_inter_sig_1 + mics_inter_sig_2;

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(steering_vec, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(steering_vec, GammaR, noise_mics_sig);

        theoretical_cor = steering_vec * steering_vec' + ...
            inter_gain_1^2 * (steering_vec_inter_1 * steering_vec_inter_1') + ...
            inter_gain_2^2 * (steering_vec_inter_2 * steering_vec_inter_2') + ...
            noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            steering_vec, theoretical_cor, noise_mics_sig);

        log_nmse_E(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_E - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_nmse_R(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_R - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_nmse_theoretical(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm);
    end
end
mean_log_nmse_E = mean(log_nmse_E, 2);
std_log_nmse_E = std(log_nmse_E, 0, 2);
mean_log_nmse_R = mean(log_nmse_R, 2);
std_log_nmse_R = std(log_nmse_R, 0, 2);
mean_log_nmse_theoretical = mean(log_nmse_theoretical, 2);
std_log_nmse_theoretical = std(log_nmse_theoretical, 0, 2);

figure(1);
hold on
errorbar(num_of_mics, mean_log_nmse_E, std_log_nmse_E)
errorbar(num_of_mics, mean_log_nmse_R, std_log_nmse_R)
errorbar(num_of_mics, mean_log_nmse_theoretical, std_log_nmse_theoretical)
title("Log NMSE Error of Estimated Signal Without Attenuation")
subtitle("Number of Samples=" + sig_length + ", SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off


%% Constant SNR, Changing Number of Microphones, With Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 0;
log_nmse_E = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_R = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_theoretical = zeros(1, length(num_of_mics));


for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    noise_gain = zeros(1, m);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    distance_trans = zeros(m, 1);
    distance_inter_1 = zeros(m, 1);
    distance_inter_2 = zeros(m, 1);
    for i = 1 : m
        distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
        distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
        distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
        target_gain_mic_i = target_gain / distance_trans(i);
        noise_gain(i) = target_gain_mic_i / 10^(SNR_dB / 20);  % Epsilon
    end
    phase_mic = distance_trans / lambda;
    phase_mic_inter_1 = distance_inter_1 / lambda;
    phase_mic_inter_2 = distance_inter_2 / lambda;
    atf_trans = exp(-1j * 2 * pi * phase_mic) ./ distance_trans;
    atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1) ./ distance_inter_1;
    atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2) ./ distance_inter_2;
    mics_sig = atf_trans * signal_target;
    mics_inter_sig_1 = atf_inter_1 * inter_sig_1;
    mics_inter_sig_2 = atf_inter_2 * inter_sig_2;
    SIR_dB_1_eff = 20 * log10 (norm(mics_sig(1, :)) / norm(mics_inter_sig_1(1, :)));
    SIR_dB_2_eff = 20 * log10 (norm(mics_sig(1, :)) / norm(mics_inter_sig_2(1, :)));
    inter_gain_1_eff = target_gain / 10^(SIR_dB_1_eff / 20);
    inter_gain_2_eff = target_gain / 10^(SIR_dB_1_eff / 20);
    atf_trans = atf_trans / atf_trans(1);
    atf_inter_1 = atf_inter_1 / atf_inter_1(1);
    atf_inter_2 = atf_inter_2 / atf_inter_2(1);
    first_mic_clean_norm = norm(mics_sig(1, :))^2;

    for monte_carlo_index = 1 : monte_carlo_num
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            added_noise(i, :) = noise_gain(i) * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise + mics_inter_sig_1 + mics_inter_sig_2;

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(atf_trans, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(atf_trans, GammaR, noise_mics_sig);

        theoretical_cor = atf_trans * atf_trans' + ...
            inter_gain_1_eff^2 * (atf_inter_1 * atf_inter_1') + ...
            inter_gain_2_eff^2 * (atf_inter_2 * atf_inter_2') + ...
            diag(noise_gain.^2);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

        log_nmse_E(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_E - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_nmse_R(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_R - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_nmse_theoretical(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm);
    end
end
mean_log_nmse_E = mean(log_nmse_E, 2);
std_log_nmse_E = std(log_nmse_E, 0, 2);
mean_log_nmse_R = mean(log_nmse_R, 2);
std_log_nmse_R = std(log_nmse_R, 0, 2);
mean_log_nmse_theoretical = mean(log_nmse_theoretical, 2);
std_log_nmse_theoretical = std(log_nmse_theoretical, 0, 2);

figure(2);
hold on
errorbar(num_of_mics, mean_log_nmse_E, std_log_nmse_E)
errorbar(num_of_mics, mean_log_nmse_R, std_log_nmse_R)
errorbar(num_of_mics, mean_log_nmse_theoretical, std_log_nmse_theoretical)
title("Log NMSE Error of Estimated Signal With Attenuation")
subtitle("Number of Samples=" + sig_length + ", SNR=" + SNR_dB + "[dB], SIR\_eff_1=" + SIR_dB_1_eff + "[dB], SIR\_eff_2=" + SIR_dB_2_eff + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off


%% Estimate ATF With Eigenvector Corresponding to The Biggest Eigenvalue, Without Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 15;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
log_mse_E = zeros(length(num_of_mics), monte_carlo_num);
log_mse_R = zeros(length(num_of_mics), monte_carlo_num);
log_mse_theoretical = zeros(length(num_of_mics), monte_carlo_num);
atf_E_similarity = zeros(length(num_of_mics), monte_carlo_num);
atf_R_similarity = zeros(length(num_of_mics), monte_carlo_num);
h_mvdr_E_similarity = zeros(length(num_of_mics), monte_carlo_num);
h_mvdr_R_similarity = zeros(length(num_of_mics), monte_carlo_num);


for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    distance_trans = zeros(m, 1);
    distance_inter_1 = zeros(m, 1);
    distance_inter_2 = zeros(m, 1);
    for i = 1 : m
        distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
        distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
        distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
    end
    phase_mic = distance_trans / lambda;
    phase_mic_inter_1 = distance_inter_1 / lambda;
    phase_mic_inter_2 = distance_inter_2 / lambda;
    atf_trans = exp(-1j * 2 * pi * phase_mic);
    atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1);
    atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2);
    mics_sig = atf_trans * signal_target;
    mics_inter_sig_1 = atf_inter_1 * inter_sig_1;
    mics_inter_sig_2 = atf_inter_2 * inter_sig_2;
    atf_trans = atf_trans / atf_trans(1);
    atf_inter_1 = atf_inter_1 / atf_inter_1(1);
    atf_inter_2 = atf_inter_2 / atf_inter_2(1);
    first_mic_clean_norm = norm(mics_sig(1, :))^2;

    for monte_carlo_index = 1 : monte_carlo_num
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise + mics_inter_sig_1 + mics_inter_sig_2;

        phi_y = noise_mics_sig * noise_mics_sig';
        [eigvec_mat_E, eigval_vec_E] = SortedEVD(phi_y);
        atf_trans_est_E = eigvec_mat_E(:, 1);
        atf_trans_est_E = atf_trans_est_E / atf_trans_est_E(1);
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(atf_trans_est_E, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
        [eigvec_mat_R, eigval_vec_R] = SortedEVD(GammaR);
        atf_trans_est_R = eigvec_mat_R(:, 1);
        atf_trans_est_R = atf_trans_est_R / atf_trans_est_R(1);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(atf_trans_est_R, GammaR, noise_mics_sig);

        theoretical_cor = atf_trans * atf_trans' + ...
            inter_gain_1^2 * (atf_inter_1 * atf_inter_1') + ...
            inter_gain_2^2 * (atf_inter_2 * atf_inter_2') + ...
            diag(noise_gain.^2);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

        log_mse_E(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_E - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_mse_R(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_R - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_mse_theoretical(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm);
        atf_E_similarity(index, monte_carlo_index) = (atf_trans' * atf_trans_est_E) / (norm(atf_trans) * norm(atf_trans_est_E));
        atf_R_similarity(index, monte_carlo_index) = (atf_trans' * atf_trans_est_R) / (norm(atf_trans) * norm(atf_trans_est_R));
        h_mvdr_E_similarity(index, monte_carlo_index) = (h_mvdr_theoretical' * h_mvdr_E) / (norm(h_mvdr_theoretical) * norm(h_mvdr_E));
        h_mvdr_R_similarity(index, monte_carlo_index) = (h_mvdr_theoretical' * h_mvdr_R) / (norm(h_mvdr_theoretical) * norm(h_mvdr_R));
    end
end
mean_log_mse_E = mean(log_mse_E, 2);
std_log_mse_E = std(log_mse_E, 0, 2);
mean_log_mse_R = mean(log_mse_R, 2);
std_log_mse_R = std(log_mse_R, 0, 2);
mean_log_mse_theoretical = mean(log_mse_theoretical, 2);
std_log_mse_theoretical = std(log_mse_theoretical, 0, 2);
mean_atf_E_similarity = mean(real(atf_E_similarity), 2);
std_atf_E_similarity = std(real(atf_E_similarity), 0, 2);
mean_atf_R_similarity = mean(real(atf_R_similarity), 2);
std_atf_R_similarity = std(real(atf_R_similarity), 0, 2);
mean_h_mvdr_E_similarity = mean(real(h_mvdr_E_similarity), 2);
std_h_mvdr_E_similarity = std(real(h_mvdr_E_similarity), 0, 2);
mean_h_mvdr_R_similarity = mean(real(h_mvdr_R_similarity), 2);
std_h_mvdr_R_similarity = std(real(h_mvdr_R_similarity), 0, 2);

figure(3);
hold on
errorbar(num_of_mics, mean_log_mse_E, std_log_mse_E)
errorbar(num_of_mics, mean_log_mse_R, std_log_mse_R)
errorbar(num_of_mics, mean_log_mse_theoretical, std_log_mse_theoretical)
title("Log NMSE Error of Estimated Signal Without Attenuation, With Estimated ATF")
subtitle("Number of Samples=" + sig_length + ", SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off

figure(4);
hold on
errorbar(num_of_mics, mean_atf_E_similarity, std_atf_E_similarity)
errorbar(num_of_mics, mean_atf_R_similarity, std_atf_R_similarity)
title("ATF Similarity Without Attenuation, $\Re\{\frac{a^H \cdot a_e}{||a|| \cdot ||a_e||}\}$", 'Interpreter', 'latex')
ylabel("Degree Similarity")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R")
hold off

figure(5);
hold on
errorbar(num_of_mics, mean_h_mvdr_E_similarity, std_h_mvdr_E_similarity)
errorbar(num_of_mics, mean_h_mvdr_R_similarity, std_h_mvdr_R_similarity)
title("h$_{MVDR}$ Similarity With Attenuation, $\Re\{\frac{h^H \cdot h_e}{||h|| \cdot ||h_e||}\}$", 'Interpreter', 'latex')
ylabel("Degree Similarity")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R")
hold off


%% Estimate ATF With Eigenvector Corresponding to The Biggest Eigenvalue, With Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 5;
log_mse_E = zeros(length(num_of_mics), monte_carlo_num);
log_mse_R = zeros(length(num_of_mics), monte_carlo_num);
log_mse_theoretical = zeros(length(num_of_mics), monte_carlo_num);
atf_E_similarity = zeros(length(num_of_mics), monte_carlo_num);
atf_R_similarity = zeros(length(num_of_mics), monte_carlo_num);
h_mvdr_E_similarity = zeros(length(num_of_mics), monte_carlo_num);
h_mvdr_R_similarity = zeros(length(num_of_mics), monte_carlo_num);


for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    noise_gain = zeros(1, m);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    distance_trans = zeros(m, 1);
    distance_inter_1 = zeros(m, 1);
    distance_inter_2 = zeros(m, 1);
    for i = 1 : m
        distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
        distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
        distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
        target_gain_mic_i = target_gain / distance_trans(i);
        noise_gain(i) = target_gain_mic_i / 10^(SNR_dB / 20);  % Epsilon
    end
    phase_mic = distance_trans / lambda;
    phase_mic_inter_1 = distance_inter_1 / lambda;
    phase_mic_inter_2 = distance_inter_2 / lambda;
    atf_trans = exp(-1j * 2 * pi * phase_mic) ./ distance_trans;
    atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1) ./ distance_inter_1;
    atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2) ./ distance_inter_2;
    mics_sig = atf_trans * signal_target;
    mics_inter_sig_1 = atf_inter_1 * inter_sig_1;
    mics_inter_sig_2 = atf_inter_2 * inter_sig_2;
    atf_trans = atf_trans / atf_trans(1);
    atf_inter_1 = atf_inter_1 / atf_inter_1(1);
    atf_inter_2 = atf_inter_2 / atf_inter_2(1);
    first_mic_clean_norm = norm(mics_sig(1, :))^2;
    
    for monte_carlo_index = 1 : monte_carlo_num
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            added_noise(i, :) = noise_gain(i) * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        noise_mics_sig = mics_sig + added_noise + mics_inter_sig_1 + mics_inter_sig_2;

        phi_y = noise_mics_sig * noise_mics_sig';
        [eigvec_mat_E, eigval_vec_E] = SortedEVD(phi_y);
        atf_trans_est_E = eigvec_mat_E(:, 1);
        atf_trans_est_E = atf_trans_est_E / atf_trans_est_E(1);
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(atf_trans_est_E, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
        [eigvec_mat_R, eigval_vec_R] = SortedEVD(GammaR);
        atf_trans_est_R = eigvec_mat_R(:, 1);
        atf_trans_est_R = atf_trans_est_R / atf_trans_est_R(1);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(atf_trans_est_R, GammaR, noise_mics_sig);

        theoretical_cor = atf_trans * atf_trans' + ...
            inter_gain_1^2 * (atf_inter_1 * atf_inter_1') + ...
            inter_gain_2^2 * (atf_inter_2 * atf_inter_2') + ...
            diag(noise_gain.^2);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

        log_mse_E(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_E - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_mse_R(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_R - mics_sig(1, :))^2 / first_mic_clean_norm);
        log_mse_theoretical(index, monte_carlo_index) = 10 * log10(norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm);
        atf_E_similarity(index, monte_carlo_index) = (atf_trans' * atf_trans_est_E) / (norm(atf_trans) * norm(atf_trans_est_E));
        atf_R_similarity(index, monte_carlo_index) = (atf_trans' * atf_trans_est_R) / (norm(atf_trans) * norm(atf_trans_est_R));
        h_mvdr_E_similarity(index, monte_carlo_index) = (h_mvdr_theoretical' * h_mvdr_E) / (norm(h_mvdr_theoretical) * norm(h_mvdr_E));
        h_mvdr_R_similarity(index, monte_carlo_index) = (h_mvdr_theoretical' * h_mvdr_R) / (norm(h_mvdr_theoretical) * norm(h_mvdr_R));
    end
end
mean_log_mse_E = mean(log_mse_E, 2);
std_log_mse_E = std(log_mse_E, 0, 2);
mean_log_mse_R = mean(log_mse_R, 2);
std_log_mse_R = std(log_mse_R, 0, 2);
mean_log_mse_theoretical = mean(log_mse_theoretical, 2);
std_log_mse_theoretical = std(log_mse_theoretical, 0, 2);
mean_atf_E_similarity = mean(real(atf_E_similarity), 2);
std_atf_E_similarity = std(real(atf_E_similarity), 0, 2);
mean_atf_R_similarity = mean(real(atf_R_similarity), 2);
std_atf_R_similarity = std(real(atf_R_similarity), 0, 2);
mean_h_mvdr_E_similarity = mean(real(h_mvdr_E_similarity), 2);
std_h_mvdr_E_similarity = std(real(h_mvdr_E_similarity), 0, 2);
mean_h_mvdr_R_similarity = mean(real(h_mvdr_R_similarity), 2);
std_h_mvdr_R_similarity = std(real(h_mvdr_R_similarity), 0, 2);

figure(6);
hold on
errorbar(num_of_mics, mean_log_mse_E, std_log_mse_E)
errorbar(num_of_mics, mean_log_mse_R, std_log_mse_R)
errorbar(num_of_mics, mean_log_mse_theoretical, std_log_mse_theoretical)
title("Log NMSE Error of Estimated Signal With Attenuation, With Estimated ATF")
subtitle("Number of Samples=" + sig_length + ", SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off

figure(7);
hold on
errorbar(num_of_mics, mean_atf_E_similarity, std_atf_E_similarity)
errorbar(num_of_mics, mean_atf_R_similarity, std_atf_R_similarity)
title("ATF Similarity With Attenuation, $\Re\{\frac{a^H \cdot a_e}{||a|| \cdot ||a_e||}\}$", 'Interpreter', 'latex')
ylabel("Degree Similarity")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R")
hold off

figure(8);
hold on
errorbar(num_of_mics, mean_h_mvdr_E_similarity, std_h_mvdr_E_similarity)
errorbar(num_of_mics, mean_h_mvdr_R_similarity, std_h_mvdr_R_similarity)
title("h$_{MVDR}$ Similarity With Attenuation, $\Re\{\frac{h^H \cdot h_e}{||h|| \cdot ||h_e||}\}$", 'Interpreter', 'latex')
ylabel("Degree Similarity")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R")
hold off

