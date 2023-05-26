%% Clears
clc; clear all; close all;

%% Randoms
trans_theta = rand * 180;
trans_theta_1 = rand * 180;
trans_theta_2 = rand * 180;


%% Definitions
m = 24;
m_lin = (0 : m - 1)';
epsilon = 1e-2;
lambda = 0.087214428857715;
delta = lambda / 2;
theta = 0 : 1e-1 : 180;
steering_vec = exp(1j * 2 * pi * (delta / lambda) .* m_lin * cos(trans_theta * pi / 180));
steering_vec_1 = exp(1j * 2 * pi * (delta / lambda) .* m_lin * cos(trans_theta_1 * pi / 180));
steering_vec_2 = exp(1j * 2 * pi * (delta / lambda) .* m_lin * cos(trans_theta_2 * pi / 180));


%% Theory Calculation Using DS and MVDR
cov_mat = steering_vec * steering_vec' + epsilon * eye(m);
p_mvdr = zeros(1, length(theta));
p_ds = zeros(1, length(theta));

for i = 1 : length(theta)
    steering_vec_theta = exp(1j * 2 * pi * delta / lambda .* m_lin * cos(theta(i) * pi / 180));
    p_mvdr(i) = 1 / (steering_vec_theta' * pinv(cov_mat) * steering_vec_theta);
    p_ds(i) = steering_vec_theta' * cov_mat * steering_vec_theta;
end

p_mvdr_full = DuplicateSpectrumFunc(p_mvdr);
p_ds_full = DuplicateSpectrumFunc(p_ds);

[~, max_theta_ind_mvdr] = max(p_mvdr_full);
theta_max_mvdr = theta(max_theta_ind_mvdr);

[~, max_theta_ind_ds] = max(p_ds_full);
theta_max_ds = theta(max_theta_ind_ds);

figure(1);
plot(theta, p_mvdr_full)
xlabel("Theta [degree]")
ylabel("MVDR")
title("Spectrum MVDR")

figure(2);
polarplot(theta / 180 * pi, p_mvdr_full)
title("Spectrum MVDR")
rlim([-40 0])
thetalim([0 180])

figure(3);
plot(theta, p_ds_full)
xlabel("Theta [degree]")
ylabel("MVDR")
title("Spectrum DS")

figure(4);
polarplot(theta / 180 * pi, p_ds_full)
title("Spectrum DS")
rlim([-20 0])
thetalim([0 180])


%% Two Transmiters
cov_mat_2 = steering_vec_1 * steering_vec_1' + 0.5 * (steering_vec_2 * steering_vec_2') + epsilon * eye(m);
p_mvdr_2 = zeros(1, length(theta));
p_ds_2 = zeros(1, length(theta));

for i = 1 : length(theta)
    steering_vec_theta = exp(1j * 2 * pi * delta / lambda .* m_lin * cos(theta(i) * pi / 180));
    p_mvdr_2(i) = 1 / (steering_vec_theta' * pinv(cov_mat_2) * steering_vec_theta);
    p_ds_2(i) = steering_vec_theta' * cov_mat_2 * steering_vec_theta;
end

p_mvdr_full_2 = DuplicateSpectrumFunc(p_mvdr_2);
p_ds_2_full = DuplicateSpectrumFunc(p_ds_2);

[~, max_theta_ind_mvdr] = max(p_mvdr_full_2);
theta_max_mvdr_2 = theta(max_theta_ind_mvdr);

[~, max_theta_ind_ds] = max(p_ds_2_full);
theta_max_ds_2 = theta(max_theta_ind_ds);

figure(1);
plot(theta, p_mvdr_full_2)
xlabel("Theta [degree]")
ylabel("MVDR")
title("Spectrum MVDR")

figure(2);
polarplot(theta / 180 * pi, p_mvdr_full_2)
title("Spectrum MVDR")
rlim([-40 0])
thetalim([0 180])

figure(3);
plot(theta, p_ds_2_full)
xlabel("Theta [degree]")
ylabel("MVDR")
title("Spectrum DS")

figure(4);
polarplot(theta / 180 * pi, p_ds_2_full)
title("Spectrum DS")
rlim([-20 0])
thetalim([0 180])


%% Functions
function SpectrumFull = DuplicateSpectrumFunc(Spectrum)
    SpectrumFull = 10 * log10(abs(Spectrum) / max(abs(Spectrum)));
end
