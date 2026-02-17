% backprop_1S1_fixed_keys_v2.m
% Corrected version: sanitizes dynamic struct field names and provides a
% compatible NaN-ignoring mean for MATLAB installations without nanmean.

clearvars; close all; clc;

%% Data
N = 201;
p = linspace(-2,2,N);
t = 1 + sin(p * pi/3);

P = reshape(p,1,[]);
T = reshape(t,1,[]);

%% Experiment settings
S1_list = [2, 6, 10, 20];
alpha_list = [0.001, 0.01, 0.05, 0.1, 0.5];
seeds = [1, 7, 42];
epochs = 5000;

results = struct();

% Helper to create a valid MATLAB struct field name
makeField = @(S1,alpha,seed) sprintf('S1_%02d_alpha_%s_seed_%d', ...
    S1, regexprep(sprintf('%.6g',alpha),'[^0-9A-Za-z]','p'), seed);

% Compatible NaN-ignoring mean function (use nanmean if available)
if exist('nanmean','file') == 2
    safe_nanmean = @(x,dim) nanmean(x,dim);
else
    safe_nanmean = @(x,dim) mean(x,dim,'omitnan');
end

%% Training loop
for sIdx = 1:numel(S1_list)
    S1 = S1_list(sIdx);
    for aIdx = 1:numel(alpha_list)
        alpha = alpha_list(aIdx);
        for seedIdx = 1:numel(seeds)
            seed = seeds(seedIdx);
            rng(seed);
            
            W1 = rand(S1,1) - 0.5;
            b1 = rand(S1,1) - 0.5;
            W2 = rand(1,S1) - 0.5;
            b2 = rand(1,1) - 0.5;
            
            mse_hist = zeros(1,epochs);
            
            for ep = 1:epochs
                Z1 = W1 * P + b1;               
                A1 = 1 ./ (1 + exp(-Z1));       
                Y  = W2 * A1 + b2;              

                E = Y - T;
                mse = mean(E.^2);
                mse_hist(ep) = mse;

                dE_dy = (2 / N) * E;

                gradW2 = dE_dy * A1.';
                gradb2 = sum(dE_dy, 2);

                dE_da1 = (W2.' * dE_dy);
                dA1_dZ1 = A1 .* (1 - A1);
                dE_dz1 = dE_da1 .* dA1_dZ1;

                gradW1 = dE_dz1 * P.';
                gradb1 = sum(dE_dz1, 2);

                W2 = W2 - alpha * gradW2;
                b2 = b2 - alpha * gradb2;
                W1 = W1 - alpha * gradW1;
                b1 = b1 - alpha * gradb1;

                if any(isnan([W1(:);W2(:);b1(:);b2(:)]))
                    mse_hist(ep+1:end) = NaN;
                    break;
                end
            end

            key = makeField(S1, alpha, seed);
            results.(key).W1 = W1;
            results.(key).b1 = b1;
            results.(key).W2 = W2;
            results.(key).b2 = b2;
            results.(key).mse_hist = mse_hist;

            Z1 = W1 * P + b1;
            A1 = 1 ./ (1 + exp(-Z1));
            Y  = W2 * A1 + b2;
            results.(key).Y = Y;
        end
    end
end

%% Visualization: MSE convergence for each S1 and alpha (averaged across seeds)
figure('Name','MSE convergence (avg over seeds)','Units','normalized','Position',[0.05 0.05 0.9 0.8]);
for sIdx = 1:numel(S1_list)
    S1 = S1_list(sIdx);
    subplot(2,2,sIdx);
    hold on;
    title(sprintf('S1 = %d', S1));
    xlabel('Epoch'); ylabel('MSE');
    for aIdx = 1:numel(alpha_list)
        alpha = alpha_list(aIdx);
        stacked = zeros(numel(seeds), epochs);
        for seedIdx = 1:numel(seeds)
            seed = seeds(seedIdx);
            key = makeField(S1, alpha, seed);
            if isfield(results, key)
                stacked(seedIdx,:) = results.(key).mse_hist;
            else
                stacked(seedIdx,:) = NaN(1,epochs);
            end
        end
        mean_mse = safe_nanmean(stacked,1);
        plot(1:epochs, mean_mse, 'LineWidth', 1.2);
    end
    legend(arrayfun(@(a) sprintf('\\alpha=%.3g',a), alpha_list, 'UniformOutput',false), 'Location','northeast');
    set(gca,'YScale','log'); grid on;
    hold off;
end

%% Visualization: final approximations for selected configs
figure('Name','Function approximation examples','Units','normalized','Position',[0.05 0.05 0.9 0.6]);
example_alphas = [0.01, 0.05, 0.1];
for sIdx = 1:numel(S1_list)
    S1 = S1_list(sIdx);
    subplot(2,2,sIdx);
    hold on;
    plot(P, T, 'k-', 'LineWidth', 1.5); % true function
    for aIdx = 1:numel(example_alphas)
        alpha = example_alphas(aIdx);
        key = makeField(S1, alpha, seeds(1)); % pick seed 1 as example
        if isfield(results, key)
            Y = results.(key).Y;
            plot(P, Y, '--', 'LineWidth', 1.2);
        end
    end
    title(sprintf('S1=%d: true vs approx (example alphas)', S1));
    xlabel('p'); ylabel('g(p)');
    legend(['target', arrayfun(@(a) sprintf('alpha=%.3g',a), example_alphas, 'UniformOutput',false)], 'Location','best');
    hold off;
end

%% Summarize final MSEs (print table)
fprintf('\nSummary of final MSEs (rows: S1, columns: alpha; values averaged over seeds)\n');
for sIdx = 1:numel(S1_list)
    S1 = S1_list(sIdx);
    row = zeros(1,numel(alpha_list));
    for aIdx = 1:numel(alpha_list)
        alpha = alpha_list(aIdx);
        vals = zeros(1,numel(seeds));
        for seedIdx = 1:numel(seeds)
            seed = seeds(seedIdx);
            key = makeField(S1, alpha, seed);
            if isfield(results,key)
                lastIdx = find(~isnan(results.(key).mse_hist),1,'last');
                if isempty(lastIdx)
                    vals(seedIdx) = NaN;
                else
                    vals(seedIdx) = results.(key).mse_hist(lastIdx);
                end
            else
                vals(seedIdx) = NaN;
            end
        end
        row(aIdx) = safe_nanmean(vals,2);
    end
    fprintf('S1=%2d : ', S1);
    fprintf('  %.3e', row);
    fprintf('\n');
end
