%% Clears
clc; clear all; close all;


%% Random Definitions
lambda = 0.087214428857715;
delta = lambda / 2;
distance_norm = 25;

sig_length = 8192;
target_gain = 1;
signal_target = randn(1, sig_length) + 1j * randn(1, sig_length);
signal_target = target_gain * signal_target / norm(signal_target);
target_angle = 35.6;
target_pos = distance_norm * [cosd(target_angle) sind(target_angle)];

SIR_dB_1 = -10;
inter_gain_1 = target_gain / 10^(SIR_dB_1 / 20);
inter_sig_1 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_1 = inter_gain_1 * inter_sig_1 / norm(inter_sig_1);
inter_angle_1 = 61.3;
inter_pos_1 = distance_norm * [cosd(inter_angle_1) sind(inter_angle_1)];

SIR_dB_2 = -10;
inter_gain_2 = target_gain / 10^(SIR_dB_2 / 20);
inter_sig_2 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_2 = inter_gain_2 * inter_sig_2 / norm(inter_sig_2);
inter_angle_2 = 13.9;
inter_pos_2 = distance_norm * [cosd(inter_angle_2) sind(inter_angle_2)];

mask = [ones(1, sig_length / 2), zeros(1, sig_length / 2)];
inter_sig_1 = inter_sig_1 .* mask;
inter_sig_2 = inter_sig_2 .* (1 - mask);

monte_carlo_num = 500;


%% Constant SNR, Changing Number of Microphones DOA, With Attenuation
theta = 0 : 1e-1 : 180;
num_of_mics = 2 : 2 : 24;
SNR_dB = 15;
noise_gain = zeros(1, length(num_of_mics));

p_mvdr_full_R = zeros(length(num_of_mics), length(theta));
p_mvdr_full_E = zeros(length(num_of_mics), length(theta));
p_ds_full_R = zeros(length(num_of_mics), length(theta));
p_ds_full_E = zeros(length(num_of_mics), length(theta));

for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    distance_trans = zeros(m, 1);
    distance_inter_1 = zeros(m, 1);
    distance_inter_2 = zeros(m, 1);
    added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
    for i = 1 : m
        distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
        distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
        distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
        target_gain_mic_i = target_gain / distance_trans(i);
        noise_gain(i) = target_gain_mic_i / 10^(SNR_dB / 20);  % Epsilon
        added_noise(i, :) = noise_gain(i) * (added_noise(i, :) / norm(added_noise(i, :)));
    end
    phase_mic = distance_trans / lambda;
    phase_mic_inter_1 = distance_inter_1 / lambda;
    phase_mic_inter_2 = distance_inter_2 / lambda;
    steering_vec = exp(-1j * 2 * pi * phase_mic) ./ distance_trans;
    steering_vec_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1) ./ distance_inter_1;
    steering_vec_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2) ./ distance_inter_2;
    mics_sig = steering_vec * signal_target;
    noise_mics_sig = mics_sig + added_noise + steering_vec_inter_1 * inter_sig_1 + steering_vec_inter_2 * inter_sig_2;
    steering_vec = steering_vec / steering_vec(1);
    steering_vec_inter_1 = steering_vec_inter_1 / steering_vec_inter_1(1);
    steering_vec_inter_2 = steering_vec_inter_2 / steering_vec_inter_2(1);

    phi_y = noise_mics_sig * noise_mics_sig';

    GammaTensor = zeros(m, m, 2);
    GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
    GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

    GammaR = RiemannianMean(GammaTensor);

%     theoretical_cor = steering_vec * steering_vec' + ...
%         inter_gain_1^2 * (steering_vec_inter_1 * steering_vec_inter_1') + ...
%         inter_gain_2^2 * (steering_vec_inter_2 * steering_vec_inter_2') + ...
%         noise_gain^2 * eye(m);

    p_mvdr_R = zeros(1, length(theta));
    p_mvdr_E = zeros(1, length(theta));
    p_ds_R = zeros(1, length(theta));
    p_ds_E = zeros(1, length(theta));

    for i = 1 : length(theta)
        steering_vec_theta = exp(1j * 2 * pi * delta / lambda .* m_lin * cos(theta(i) * pi / 180)) ./ distance_trans;
%         steering_vec_theta = steering_vec_theta / steering_vec_theta(1);
        p_mvdr_R(i) = 1 / (steering_vec_theta' * pinv(GammaR) * steering_vec_theta);
        p_mvdr_E(i) = 1 / (steering_vec_theta' * pinv(phi_y) * steering_vec_theta);
        p_ds_R(i) = steering_vec_theta' * GammaR * steering_vec_theta;
        p_ds_E(i) = steering_vec_theta' * phi_y * steering_vec_theta;
    end

    p_mvdr_full_R(index, :) = DuplicateSpectrumFunc(p_mvdr_R);
    p_ds_full_R(index, :) = DuplicateSpectrumFunc(p_ds_R);

    [~, max_theta_ind_mvdr_R] = max(p_mvdr_full_R(index, :));
    theta_max_mvdr_R = theta(max_theta_ind_mvdr_R);
    [~, max_theta_ind_ds_R] = max(p_ds_full_R(index, :));
    theta_max_ds_R = theta(max_theta_ind_ds_R);

    p_mvdr_full_E(index, :) = DuplicateSpectrumFunc(p_mvdr_E);
    p_ds_full_E(index, :) = DuplicateSpectrumFunc(p_ds_E);

    [~, max_theta_ind_mvdr_E] = max(p_mvdr_full_E(index, :));
    theta_max_mvdr_E = theta(max_theta_ind_mvdr_E);
    [~, max_theta_ind_ds_E] = max(p_ds_full_E(index, :));
    theta_max_ds_E = theta(max_theta_ind_ds_E);
end

fig = figure(1);
fig.WindowState = 'maximized';
filename = "DOA-Animation-MVDR-With-Attenuation-2.gif";  % Specify the output file name
target_angle_rad = target_angle * pi / 180;
inter_angle_1_rad = inter_angle_1 * pi / 180;
inter_angle_2_rad = inter_angle_2 * pi / 180;
for index = 1 : length(num_of_mics)
    polarplot(theta / 180 * pi, p_mvdr_full_R(index, :), 'LineWidth', 3)
    hold on
    polarplot(theta / 180 * pi, p_mvdr_full_E(index, :), ':', 'LineWidth', 3)
    polarplot([target_angle_rad; target_angle_rad], [-20; 0], 'LineWidth', 2, 'Color', 'black')
    polarplot([inter_angle_1_rad; inter_angle_1_rad], [-20; 0], '-.', 'LineWidth', 2, 'Color', 'black')
    polarplot([inter_angle_1_rad; inter_angle_2_rad], [-20; 0], '-.', 'LineWidth', 2, 'Color', 'black')
    title("MVDR Spectrum, Microphones: " + num_of_mics(index))
    subtitle("SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
    hold off
    rlim([-20 0])
    thetalim([0 180])
    legend("Sampling \Gamma_R", ...
        "Sampling \Gamma_E", ...
        "Target Tranmitter", "Interference")
    pause(1)
    drawnow
    frame = getframe(fig);
    im = frame2im(frame);
    [A, map] = rgb2ind(im, 256);
    if index == 1
        imwrite(A, map, filename, "gif", "LoopCount", Inf, "DelayTime", 1);
    else
        imwrite(A, map, filename, "gif", "WriteMode", "append", "DelayTime", 1);
    end
end

fig = figure(2);
fig.WindowState = 'maximized';
filename = "DOA-Animation-DS-With-Attenuation-2.gif";  % Specify the output file name
for index = 1 : length(num_of_mics)
    polarplot(theta / 180 * pi, p_ds_full_R(index, :), 'LineWidth', 3)
    hold on
    polarplot(theta / 180 * pi, p_ds_full_E(index, :), ':', 'LineWidth', 3)
    polarplot([target_angle_rad; target_angle_rad], [-20; 0], 'LineWidth', 2, 'Color', 'black')
    polarplot([inter_angle_1_rad; inter_angle_1_rad], [-20; 0], '-.', 'LineWidth', 2, 'Color', 'black')
    polarplot([inter_angle_1_rad; inter_angle_2_rad], [-20; 0], '-.', 'LineWidth', 2, 'Color', 'black')
    title("DS Spectrum, Microphones: " + num_of_mics(index))
    subtitle("SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
    hold off
    rlim([-20 0])
    thetalim([0 180])
    legend("Sampling \Gamma_R", ...
        "Sampling \Gamma_E", ...
        "Target Tranmitter", "Interference")
    pause(1)
    drawnow
    frame = getframe(fig);
    im = frame2im(frame);
    [A, map] = rgb2ind(im, 256);
    if index == 1
        imwrite(A, map, filename, "gif", "LoopCount", Inf, "DelayTime", 1);
    else
        imwrite(A, map, filename, "gif", "WriteMode", "append", "DelayTime", 1);
    end
end


%% Constant SNR, Changing Number of Microphones, Without Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 15;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
mse_E = zeros(1, length(num_of_mics));
mse_R = zeros(1, length(num_of_mics));
mse_theoretical = zeros(1, length(num_of_mics));

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(num_of_mics)
        m = num_of_mics(index);
        m_lin = (0 : m - 1)';
        mics_pos_mat = [m_lin * delta, zeros(m, 1)];
        distance_trans = zeros(m, 1);
        distance_inter_1 = zeros(m, 1);
        distance_inter_2 = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
            distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
            distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        phase_mic = distance_trans / lambda;
        phase_mic_inter_1 = distance_inter_1 / lambda;
        phase_mic_inter_2 = distance_inter_2 / lambda;
        steering_vec = exp(-1j * 2 * pi * phase_mic);
        steering_vec_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1);
        steering_vec_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2);
        mics_sig = steering_vec * signal_target;
        noise_mics_sig = mics_sig + added_noise + steering_vec_inter_1 * inter_sig_1 + steering_vec_inter_2 * inter_sig_2;
        steering_vec = steering_vec / steering_vec(1);
        steering_vec_inter_1 = steering_vec_inter_1 / steering_vec_inter_1(1);
        steering_vec_inter_2 = steering_vec_inter_2 / steering_vec_inter_2(1);

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

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
        mse_E(index) = mse_E(index) + norm(estimated_sig_E - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_R(index) = mse_R(index) + norm(estimated_sig_R - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm;
    end
end
mse_E = mse_E / monte_carlo_num;
mse_R = mse_R / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;

figure(1);
hold on
plot(num_of_mics, 10 * log10(mse_E))
plot(num_of_mics, 10 * log10(mse_R))
plot(num_of_mics, 10 * log10(mse_theoretical))
title("Log NMSE Error of Estimated Signal")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off


%% Constant SNR, Changing Number of Microphones, With Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 5;
log_nmse_E = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_R = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_theoretical = zeros(1, length(num_of_mics));

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(num_of_mics)
        m = num_of_mics(index);
        noise_gain = zeros(1, m);
        m_lin = (0 : m - 1)';
        mics_pos_mat = [m_lin * delta, zeros(m, 1)];
        distance_trans = zeros(m, 1);
        distance_inter_1 = zeros(m, 1);
        distance_inter_2 = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
            distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
            distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
            target_gain_mic_i = target_gain / distance_trans(i);
            noise_gain(i) = target_gain_mic_i / 10^(SNR_dB / 20);  % Epsilon
            added_noise(i, :) = noise_gain(i) * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        phase_mic = distance_trans / lambda;
        phase_mic_inter_1 = distance_inter_1 / lambda;
        phase_mic_inter_2 = distance_inter_2 / lambda;
        atf_trans = exp(-1j * 2 * pi * phase_mic) ./ distance_trans;
        atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1) ./ distance_inter_1;
        atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2) ./ distance_inter_2;
        mics_sig = atf_trans * signal_target;
        noise_mics_sig = mics_sig + added_noise + atf_inter_1 * inter_sig_1 + atf_inter_2 * inter_sig_2;
        atf_trans = atf_trans / atf_trans(1);
        atf_inter_1 = atf_inter_1 / atf_inter_1(1);
        atf_inter_2 = atf_inter_2 / atf_inter_2(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(atf_trans, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(atf_trans, GammaR, noise_mics_sig);

        theoretical_cor = atf_trans * atf_trans' + ...
            inter_gain_1^2 * (atf_inter_1 * atf_inter_1') + ...
            inter_gain_2^2 * (atf_inter_2 * atf_inter_2') + ...
            diag(noise_gain.^2);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
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
subtitle("SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off


%% Estimate ATF With Eigenvector Corresponding to The Biggest Eigenvalue, Without Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 15;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon
log_nmse_E = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_R = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_theoretical = zeros(length(num_of_mics), monte_carlo_num);
atf_E_similarity = zeros(1, length(num_of_mics));
atf_R_similarity = zeros(1, length(num_of_mics));

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(num_of_mics)
        m = num_of_mics(index);
        m_lin = (0 : m - 1)';
        mics_pos_mat = [m_lin * delta, zeros(m, 1)];
        distance_trans = zeros(m, 1);
        distance_inter_1 = zeros(m, 1);
        distance_inter_2 = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
            distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
            distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
            added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        phase_mic = distance_trans / lambda;
        phase_mic_inter_1 = distance_inter_1 / lambda;
        phase_mic_inter_2 = distance_inter_2 / lambda;
        atf_trans = exp(-1j * 2 * pi * phase_mic);
        atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1);
        atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2);
        mics_sig = atf_trans * signal_target;
        noise_mics_sig = mics_sig + added_noise + atf_inter_1 * inter_sig_1 + atf_inter_2 * inter_sig_2;
        atf_trans = atf_trans / atf_trans(1);
        atf_inter_1 = atf_inter_1 / atf_inter_1(1);
        atf_inter_2 = atf_inter_2 / atf_inter_2(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [eigvec_mat_E, eigval_vec_E] = SortedEVD(phi_y);
        atf_trans_est_E = eigvec_mat_E(:, 1);
%         [atf_trans_est_E, ~] = eigs(phi_y, 1);
        atf_trans_est_E = atf_trans_est_E / atf_trans_est_E(1);
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(atf_trans_est_E, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
%         [eigvec_mat_R, eigval_vec_R] = SortedEVD(GammaR);
%         atf_trans_est_R = eigvec_mat_R(:, 1);
        [atf_trans_est_R, ~] = eigs(GammaR, 1);
        atf_trans_est_R = atf_trans_est_R / atf_trans_est_R(1);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(atf_trans_est_R, GammaR, noise_mics_sig);

        theoretical_cor = atf_trans * atf_trans' + ...
            inter_gain_1^2 * (atf_inter_1 * atf_inter_1') + ...
            inter_gain_2^2 * (atf_inter_2 * atf_inter_2') + ...
            diag(noise_gain.^2);
%         [eigvec_mat_T, eigval_vec_T] = SortedEVD(theoretical_cor);
%         atf_trans_est_T = eigvec_mat_T(:, 1);
%         atf_trans_est_T = atf_trans_est_T / atf_trans_est_T(1);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

%         first_mic_clean_norm = norm(mics_sig(1, :))^2;
        log_nmse_E(index, monte_carlo_index) = norm(estimated_sig_E - mics_sig(1, :))^2 / sig_length;
        log_nmse_R(index, monte_carlo_index) = norm(estimated_sig_R - mics_sig(1, :))^2 / sig_length;
        log_nmse_theoretical(index, monte_carlo_index) = norm(estimated_sig_theoretical - mics_sig(1, :))^2 / sig_length;
        atf_E_similarity(index) = atf_E_similarity(index) + ...
            (atf_trans' * atf_trans_est_E) / (norm(atf_trans) * norm(atf_trans_est_E));
        atf_R_similarity(index) = atf_R_similarity(index) + ...
            (atf_trans' * atf_trans_est_R) / (norm(atf_trans) * norm(atf_trans_est_R));
    end
end
mean_log_mse_E = mean(log_nmse_E, 2);
std_log_mse_E = std(log_nmse_E, 0, 2);
mean_log_mse_R = mean(log_nmse_R, 2);
std_log_mse_R = std(log_nmse_R, 0, 2);
mean_log_mse_theoretical = mean(log_nmse_theoretical, 2);
std_log_mse_theoretical = std(log_nmse_theoretical, 0, 2);
atf_E_similarity = atf_E_similarity / monte_carlo_num;
atf_R_similarity = atf_R_similarity / monte_carlo_num;

figure(3);
hold on
errorbar(num_of_mics, mean_log_mse_E, std_log_mse_E)
errorbar(num_of_mics, mean_log_mse_R, std_log_mse_R)
errorbar(num_of_mics, mean_log_mse_theoretical, std_log_mse_theoretical)
title("Log NMSE Error of Estimated Signal Without Attenuation")
subtitle("SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off

figure(4);
hold on
plot(num_of_mics, atf_E_similarity)
plot(num_of_mics, atf_R_similarity)
title("ATF Similarity Without Attenuation")
ylabel("Degree Similarity")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R")
hold off


%% Estimate ATF With Eigenvector Corresponding to The Biggest Eigenvalue, With Attenuation
num_of_mics = 8 : 2 : 24;
SNR_dB = 15;
log_nmse_E = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_R = zeros(length(num_of_mics), monte_carlo_num);
log_nmse_theoretical = zeros(length(num_of_mics), monte_carlo_num);
atf_E_similarity = zeros(1, length(num_of_mics));
atf_R_similarity = zeros(1, length(num_of_mics));

for monte_carlo_index = 1 : monte_carlo_num
    for index = 1 : length(num_of_mics)
        m = num_of_mics(index);
        noise_gain = zeros(1, m);
        m_lin = (0 : m - 1)';
        mics_pos_mat = [m_lin * delta, zeros(m, 1)];
        distance_trans = zeros(m, 1);
        distance_inter_1 = zeros(m, 1);
        distance_inter_2 = zeros(m, 1);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
            distance_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1);
            distance_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2);
            target_gain_mic_i = target_gain / distance_trans(i);
            noise_gain(i) = target_gain_mic_i / 10^(SNR_dB / 20);  % Epsilon
            added_noise(i, :) = noise_gain(i) * (added_noise(i, :) / norm(added_noise(i, :)));
        end
        phase_mic = distance_trans / lambda;
        phase_mic_inter_1 = distance_inter_1 / lambda;
        phase_mic_inter_2 = distance_inter_2 / lambda;
        atf_trans = exp(-1j * 2 * pi * phase_mic) ./ distance_trans;
        atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1) ./ distance_inter_1;
        atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2) ./ distance_inter_2;
        mics_sig = atf_trans * signal_target;
        noise_mics_sig = mics_sig + added_noise + atf_inter_1 * inter_sig_1 + atf_inter_2 * inter_sig_2;
        atf_trans = atf_trans / atf_trans(1);
        atf_inter_1 = atf_inter_1 / atf_inter_1(1);
        atf_inter_2 = atf_inter_2 / atf_inter_2(1);

        phi_y = noise_mics_sig * noise_mics_sig';
        [eigvec_mat_E, eigval_vec_E] = SortedEVD(phi_y);
        atf_trans_est_E = eigvec_mat_E(:, 1);
%         [atf_trans_est_E, ~] = eigs(phi_y, 1);
        atf_trans_est_E = atf_trans_est_E / atf_trans_est_E(1);
        [h_mvdr_E, estimated_sig_E] = MvdrCoefficients(atf_trans_est_E, phi_y, noise_mics_sig);

        GammaTensor = zeros(m, m, 2);
        GammaTensor(:, :, 1) = noise_mics_sig(:, 1:sig_length / 2) * noise_mics_sig(:, 1:sig_length / 2)';
        GammaTensor(:, :, 2) = noise_mics_sig(:, 1 + sig_length / 2:end) * noise_mics_sig(:, 1 + sig_length / 2:end)';

        GammaR = RiemannianMean(GammaTensor);
%         [eigvec_mat_R, eigval_vec_R] = SortedEVD(GammaR);
%         atf_trans_est_R = eigvec_mat_R(:, 1);
        [atf_trans_est_R, ~] = eigs(GammaR, 1);
        atf_trans_est_R = atf_trans_est_R / atf_trans_est_R(1);
        [h_mvdr_R, estimated_sig_R] = MvdrCoefficients(atf_trans_est_R, GammaR, noise_mics_sig);

        theoretical_cor = atf_trans * atf_trans' + ...
            inter_gain_1^2 * (atf_inter_1 * atf_inter_1') + ...
            inter_gain_2^2 * (atf_inter_2 * atf_inter_2') + ...
            diag(noise_gain.^2);
%         [eigvec_mat_T, eigval_vec_T] = SortedEVD(theoretical_cor);
%         atf_trans_est_T = eigvec_mat_T(:, 1);
%         atf_trans_est_T = atf_trans_est_T / atf_trans_est_T(1);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

%         first_mic_clean_norm = norm(mics_sig(1, :))^2;
        log_nmse_E(index, monte_carlo_index) = norm(estimated_sig_E - mics_sig(1, :))^2 / sig_length;
        log_nmse_R(index, monte_carlo_index) = norm(estimated_sig_R - mics_sig(1, :))^2 / sig_length;
        log_nmse_theoretical(index, monte_carlo_index) = norm(estimated_sig_theoretical - mics_sig(1, :))^2 / sig_length;
        atf_E_similarity(index) = atf_E_similarity(index) + ...
            (atf_trans' * atf_trans_est_E) / (norm(atf_trans) * norm(atf_trans_est_E));
        atf_R_similarity(index) = atf_R_similarity(index) + ...
            (atf_trans' * atf_trans_est_R) / (norm(atf_trans) * norm(atf_trans_est_R));
    end
end
mean_log_mse_E = mean(log_nmse_E, 2);
std_log_mse_E = std(log_nmse_E, 0, 2);
mean_log_mse_R = mean(log_nmse_R, 2);
std_log_mse_R = std(log_nmse_R, 0, 2);
mean_log_mse_theoretical = mean(log_nmse_theoretical, 2);
std_log_mse_theoretical = std(log_nmse_theoretical, 0, 2);
atf_E_similarity = atf_E_similarity / monte_carlo_num;
atf_R_similarity = atf_R_similarity / monte_carlo_num;

figure(3);
hold on
errorbar(num_of_mics, mean_log_mse_E, std_log_mse_E)
errorbar(num_of_mics, mean_log_mse_R, std_log_mse_R)
errorbar(num_of_mics, mean_log_mse_theoretical, std_log_mse_theoretical)
title("Log NMSE Error of Estimated Signal With Attenuation")
subtitle("SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R", "Population \Gamma")
hold off

figure(4);
hold on
plot(num_of_mics, atf_E_similarity)
plot(num_of_mics, atf_R_similarity)
title("ATF Similarity With Attenuation")
ylabel("Degree Similarity")
xlabel("Number of Microphones")
legend("Sampling \Gamma_E", "Sampling \Gamma_R")
hold off


%% Functions
function SpectrumFull = DuplicateSpectrumFunc(Spectrum)
    SpectrumFull = 10 * log10(abs(Spectrum) / max(abs(Spectrum)));
end

