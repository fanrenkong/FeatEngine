addpath(genpath(cd));
format long
format compact
clc

rand('state', sum(100*clock));
randn('state', sum(100*clock));
%% Settings
N = 100; % Population size
tradeOff = 0.8; % Tradeoff factor
nfevalmax = 1E3; % Max function evaluation
runTimes = 30; % Total run times

%% Run
fprintf('BEGIN TO RUN [%s] IN [%s] DATASET WITH [%s] AS CLASSIFIER\n', ...
        func2str(EC), class(Problem), func2str(CA));
for run = 1 : runTimes
    %% Classification Algorithm
    CA = @fitcknn; 

    %% Problem
    Problem = NineTumor(nfevalmax);

    %% Evolutionary Computation
    EC = @SMMOEAFS;
    
    [result, CPUTime] = EC(N, Problem, CA, tradeOff);
    fprintf('------------------------ RUN [%d / %d] RESULTS --------------------------------\n', ...
        run, runTimes);
    fprintf('SOLUTION COUNT: %d\n', result.n);
    fprintf('AVERAGE FEATURES COUNT: %.4f/%d\n', mean(sum(result.solutions, 2)), Problem.D);
    fprintf('AVERAGE ERROR RATE: %.4f\n', mean(result.objs(:,1)));
end
%% Display
clearvars -except result CPUTime