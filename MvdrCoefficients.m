function [MVDR_coeff, estimated_sig] = MvdrCoefficients(steering_vec, cov, noise_sig)
    MVDR_coeff = pinv(cov) * steering_vec / (steering_vec' * pinv(cov) * steering_vec);
    estimated_sig = MVDR_coeff' * noise_sig;
end
