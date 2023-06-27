%% Clears
clc; clear all; close all;


%% Random Definitions
lambda = 0.087214428857715;
delta = lambda / 2;
distance_norm = 25;

sig_length = 1024;
target_gain = 1;
signal_target = randn(1, sig_length) + 1j * randn(1, sig_length);
signal_target = target_gain * signal_target / norm(signal_target);
target_angle = 35.6;
target_pos = distance_norm * [cosd(target_angle) sind(target_angle)];

SIR_dB_1 = -20;
inter_gain_1 = target_gain / 10^(SIR_dB_1 / 20);
inter_sig_1 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_1 = inter_gain_1 * inter_sig_1 / norm(inter_sig_1);
inter_angle_1 = 41.3;
inter_pos_1 = distance_norm * [cosd(inter_angle_1) sind(inter_angle_1)];

SIR_dB_2 = -18;
inter_gain_2 = target_gain / 10^(SIR_dB_2 / 20);
inter_sig_2 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_2 = inter_gain_2 * inter_sig_2 / norm(inter_sig_2);
inter_angle_2 = 26.5;
inter_pos_2 = distance_norm * [cosd(inter_angle_2) sind(inter_angle_2)];

mask = [ones(1, sig_length / 2), zeros(1, sig_length / 2)];
inter_sig_1 = inter_sig_1 .* mask;
inter_sig_2 = inter_sig_2 .* (1 - mask);

monte_carlo_num = 500;


%% Constant SNR, Changing Number of Microphones DOA, With Attenuation
theta = 0 : 1e-1 : 180;
num_of_mics = 6 : 2 : 24;
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
        steering_vec_theta = steering_vec_theta / steering_vec_theta(1);
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
filename = "DOA-Animation-MVDR-With-Attenuation-1.gif";  % Specify the output file name
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
    subtitle("Number of Samples=" + sig_length + ", SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
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
filename = "DOA-Animation-DS-With-Attenuation-1.gif";  % Specify the output file name
for index = 1 : length(num_of_mics)
    polarplot(theta / 180 * pi, p_ds_full_R(index, :), 'LineWidth', 3)
    hold on
    polarplot(theta / 180 * pi, p_ds_full_E(index, :), ':', 'LineWidth', 3)
    polarplot([target_angle_rad; target_angle_rad], [-20; 0], 'LineWidth', 2, 'Color', 'black')
    polarplot([inter_angle_1_rad; inter_angle_1_rad], [-20; 0], '-.', 'LineWidth', 2, 'Color', 'black')
    polarplot([inter_angle_1_rad; inter_angle_2_rad], [-20; 0], '-.', 'LineWidth', 2, 'Color', 'black')
    title("DS Spectrum, Microphones: " + num_of_mics(index))
    subtitle("Number of Samples=" + sig_length + ", SNR=" + SNR_dB + "[dB], SIR_1=" + SIR_dB_1 + "[dB], SIR_2=" + SIR_dB_2 + "[dB]")
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


%% Functions
function SpectrumFull = DuplicateSpectrumFunc(Spectrum)
    SpectrumFull = 10 * log10(abs(Spectrum) / max(abs(Spectrum)));
end
