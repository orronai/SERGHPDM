%% Clears
clc; clear all; close all;


%% Random Definitions
lambda = 0.087214428857715;
delta = lambda / 2;

sig_length = 8192;
target_gain = 1;
signal_target = randn(1, sig_length) + 1j * randn(1, sig_length);
signal_target = target_gain * signal_target / norm(signal_target);
target_pos = [19 30];

SIR_dB_1 = -10;
inter_gain_1 = target_gain / 10^(SIR_dB_1 / 20);
inter_sig_1 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_1 = inter_gain_1 * inter_sig_1 / norm(inter_sig_1);
inter_pos_1 = [46 11];

SIR_dB_2 = -10;
inter_gain_2 = target_gain / 10^(SIR_dB_2 / 20);
inter_sig_2 = randn(1, sig_length) + 1j * randn(1, sig_length);
inter_sig_2 = inter_gain_2 * inter_sig_2 / norm(inter_sig_2);
inter_pos_2 = [5 45];

mask = [ones(1, sig_length / 2), zeros(1, sig_length / 2)];
inter_sig_1 = inter_sig_1 .* mask;
inter_sig_2 = inter_sig_2 .* (1 - mask);

monte_carlo_num = 500;


%% Constant SNR, Changing Number of Microphones DOA
theta = 0 : 1e-1 : 180;
num_of_mics = 2 : 2 : 20;
SNR_dB = 5;
noise_gain = target_gain / 10^(SNR_dB / 20);  % Epsilon

for index = 1 : length(num_of_mics)
    m = num_of_mics(index);
    m_lin = (0 : m - 1)';
    mics_pos_mat = [m_lin * delta, zeros(m, 1)];
    phase_mic = zeros(m, 1);
    phase_mic_inter_1 = zeros(m, 1);
    phase_mic_inter_2 = zeros(m, 1);
    added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
    for i = 1 : m
        phase_mic(i) = norm(mics_pos_mat(i, :) - target_pos) / lambda;
        added_noise(i, :) = noise_gain * (added_noise(i, :) / norm(added_noise(i, :)));
        phase_mic_inter_1(i) = norm(mics_pos_mat(i, :) - inter_pos_1) / lambda;
        phase_mic_inter_2(i) = norm(mics_pos_mat(i, :) - inter_pos_2) / lambda;
    end
    steering_vec = exp(-1j * 2 * pi * phase_mic);
    steering_vec_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1);
    steering_vec_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2);
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
    
    for i = 1 : length(theta)
        steering_vec_theta = exp(1j * 2 * pi * delta / lambda .* m_lin * cos(theta(i) * pi / 180));
        p_mvdr_R(i) = 1 / (steering_vec_theta' * pinv(GammaR) * steering_vec_theta);
        p_mvdr_E(i) = 1 / (steering_vec_theta' * pinv(phi_y) * steering_vec_theta);
    end
    
    p_mvdr_full_R = DuplicateSpectrumFunc(p_mvdr_R);
    
    [~, max_theta_ind_mvdr_R] = max(p_mvdr_full_R);
    theta_max_mvdr_R = theta(max_theta_ind_mvdr_R);

    p_mvdr_full_E = DuplicateSpectrumFunc(p_mvdr_E);
    
    [~, max_theta_ind_mvdr_E] = max(p_mvdr_full_E);
    theta_max_mvdr_E = theta(max_theta_ind_mvdr_E);

    figure;
    polarplot(theta / 180 * pi, p_mvdr_full_R)
    title("Spectrum MVDR Riemmanian, Microphones: " + m)
    rlim([-20 0])
    thetalim([0 180])

    figure;
    polarplot(theta / 180 * pi, p_mvdr_full_E)
    title("Spectrum MVDR Euclidian, Microphones: " + m)
    rlim([-20 0])
    thetalim([0 180])
end


%% Constant SNR, Changing Number of Microphones
num_of_mics = 2 : 2 : 20;
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
        distance_inter_2 = zeros(m, 2);
        added_noise = randn(m, sig_length) + 1j * randn(m, sig_length);
        for i = 1 : m
            distance_trans(i) = norm(mics_pos_mat(i, :) - target_pos);
            distance_inter_1 = norm(mics_pos_mat(i, :) - inter_pos_1);
            distance_inter_2 = norm(mics_pos_mat(i, :) - inter_pos_2);
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
        eig_t = eig(theoretical_cor);
        eig_e = eig(phi_y);
        eig_r = eig(GammaR);
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
legend("Empirical Euclidian Correlation Matrix", "Empirical Riemmanian Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Constant SNR, Changing Number of Microphones
num_of_mics = 2 : 2 : 20;
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
            noise_gain^2 * eye(m);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

        first_mic_clean_norm = norm(mics_sig(1, :))^2;
        mse_E(index) = mse_E(index) + norm(estimated_sig_E - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_R(index) = mse_R(index) + norm(estimated_sig_R - mics_sig(1, :))^2 / first_mic_clean_norm;
        mse_theoretical(index) = mse_theoretical(index) + ...
            norm(estimated_sig_theoretical - mics_sig(1, :))^2 / first_mic_clean_norm;
        eig_t = eig(theoretical_cor);
        eig_e = eig(phi_y);
        eig_r = eig(GammaR);
    end
end
mse_E = mse_E / monte_carlo_num;
mse_R = mse_R / monte_carlo_num;
mse_theoretical = mse_theoretical / monte_carlo_num;

figure(2);
hold on
plot(num_of_mics, 10 * log10(mse_E))
plot(num_of_mics, 10 * log10(mse_R))
plot(num_of_mics, 10 * log10(mse_theoretical))
title("Log NMSE Error of Estimated Signal With Attenuation")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Empirical Euclidian Correlation Matrix", "Empirical Riemmanian Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Estimate ATF With Eigenvector Corresponding to The Biggest Eigenvalue
num_of_mics = 2 : 2 : 20;
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
        atf_trans = exp(-1j * 2 * pi * phase_mic) ./ distance_trans;
        atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1) ./ distance_inter_1;
        atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2) ./ distance_inter_2;
%         atf_trans = exp(-1j * 2 * pi * phase_mic);
%         atf_inter_1 = exp(-1j * 2 * pi * phase_mic_inter_1);
%         atf_inter_2 = exp(-1j * 2 * pi * phase_mic_inter_2);
        mics_sig = atf_trans * signal_target;
        noise_mics_sig = mics_sig + added_noise + atf_inter_1 * inter_sig_1 + atf_inter_2 * inter_sig_2;
        atf_trans = atf_trans / atf_trans(1);
        atf_inter_1 = atf_inter_1 / atf_inter_1(1);
        atf_inter_2 = atf_inter_2 / atf_inter_2(1);

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
            noise_gain^2 * eye(m);
%         [eigvec_mat_T, eigval_vec_T] = SortedEVD(theoretical_cor);
%         atf_trans_est_T = eigvec_mat_T(:, 1);
%         atf_trans_est_T = atf_trans_est_T / atf_trans_est_T(1);
        [h_mvdr_theoretical, estimated_sig_theoretical] = MvdrCoefficients(...
            atf_trans, theoretical_cor, noise_mics_sig);

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

figure(3);
hold on
plot(num_of_mics, 10 * log10(mse_E))
plot(num_of_mics, 10 * log10(mse_R))
plot(num_of_mics, 10 * log10(mse_theoretical))
title("Log NMSE Error of Estimated Signal With Attenuation and Estimated ATF")
ylabel("Log NMSE")
xlabel("Number of Microphones")
legend("Empirical Euclidian Correlation Matrix", "Empirical Riemmanian Correlation Matrix", "Theoretical Correlation Matrix")
hold off


%% Functions
function SpectrumFull = DuplicateSpectrumFunc(Spectrum)
    SpectrumFull = 10 * log10(abs(Spectrum) / max(abs(Spectrum)));
end

