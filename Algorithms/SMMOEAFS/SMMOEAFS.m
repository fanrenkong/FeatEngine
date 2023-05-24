% A Steering-Matrix-Based Multiobjective Evolutionary Algorithm for High-Dimensional Feature Selection
% Programmed by Kevin Kong
function [result, CPUTime] = SMMOEAFS(N, Problem, CA, tradeOff)
    tic;
    Problem.nfeval = 0;
    % Threshold used to judge if a feature is selected in real encoding
    theta = 0;
    % Lower bound of decision variable
    lb = -ones(1, Problem.D); 
    % Upper bound of decision variable
    ub = ones(1, Problem.D);
    
    Xmax = ub;
    Xmin = lb;
    %% Parameters
    gamma = 0.1; % attenuation factor
    %% Initialize population
    population = realInit(N, Xmin, Xmax);
    %% Evaluate
    fitness = zeros(N, 2);
    x = population > theta;
    fitness(:, 1) = Problem.CalErr(x, CA);
    fitness(:, 2) = Problem.CalFeatsCount(x, true);
    Problem.nfeval = Problem.nfeval + N; 
    %% Initialize steering matrix
    SM = initSM(Problem, fitness, N);
    
    while Problem.NotTerminated()
        %% Population in generation t 
        Pt = population;
        [FrontNo, ~] = NDSort(fitness, 0, N);
        %% Dimensionality Reduction 
        % Calculate flip probability FP
        FP = exp(-gamma * Problem.nfeval) ./ (1 + exp(SM - mean(SM, 2)));
        r = rand(sum(FrontNo ~= 1), Problem.D);
        population(FrontNo ~= 1, :) = -(FP(FrontNo ~= 1, :) > r & population(FrontNo ~= 1, :) > theta) .* population(FrontNo ~= 1, :) ...
            + (FP(FrontNo ~= 1, :) <= r & population(FrontNo ~= 1, :) <= theta) .* population(FrontNo ~= 1, :);
        
        %% Individual Repair
        % Use elite individuals to repair non-elite individuals
        elite = find(FrontNo == 1);
        eliteN = length(elite);
        % average error rate of generation t 
        eliteErr = mean(fitness(elite, 1)); 
        randomEliteIndx = elite(randi(eliteN, 1, sum(FrontNo ~= 1)));
        r = rand(sum(FrontNo ~= 1), Problem.D);
        rho = abs(SM(randomEliteIndx, :) - SM(FrontNo~=1, :)) ./ SM(randomEliteIndx, :);
        population(FrontNo ~= 1, :) = (SM(FrontNo~=1, :) >= SM(randomEliteIndx, :) & population(randomEliteIndx, :) > theta) .* population(FrontNo~=1, :) ...
            + (SM(FrontNo~=1, :) < SM(randomEliteIndx, :) & rho >= r) .* population(randomEliteIndx, :) ...
            + (SM(FrontNo~=1, :) < SM(randomEliteIndx, :) & rho < r) .* population(FrontNo ~= 1, :);
        %% Evolution Algorithm
        population = GA(population, Xmin, Xmax);
        %% Evaluate
        fitness = zeros(N, 2);
        x = population > theta;
        fitness(:, 1) = Problem.CalErr(x, CA);
        fitness(:, 2) = Problem.CalFeatsCount(x, true);
        Problem.nfeval = Problem.nfeval + N; 
        %% Population in next generation t+1
        Pt_1 = population;
        %% Steering Matrix Update
        [FrontNoNext, ~] = NDSort(fitness, 0, N);
        % average error rate of generation t+1
        eliteNextErr = mean(fitness(FrontNoNext == 1, 1)); 
        if eliteNextErr < eliteErr
            % Reward(Improvement)
            % Update Steering Matrix
            SM = updateSM(SM, Pt, Pt_1, theta);
        else
            % Punish(No improvement!)
            % Reinitialize Steering Matrix
            SM = initSM(Problem, fitness, N);
        end
    end
    CPUTime = toc;
    [FrontNo, ~] = NDSort(fitness, 0, N);
    F1 = FrontNo == 1;
    result.n = sum(F1);
    result.solutions = population(F1, :) > theta;
    result.objs = fitness(F1, :);
end
%% Initialize steering matrix
function SM = initSM(Problem, fitness, N)
    %% Calculate Feature Importance(FI)
    FI = zeros(1, Problem.D);
    for d = 1 : Problem.D
        FI(d) = featureImportance(Problem.all_features(Problem.trainIdx, d), Problem.labels);
    end
    FI = FI ./ sum(FI);
    %% Calculate Individual Importance(II)
    [FrontNo, ~] = NDSort(fitness, 0, N);
    II = sum(FrontNo > FrontNo.', 2);
    II = II ./ N;
    %% Calculate SM
    SM = II * FI;
end

%% Update steering matrix
function SM = updateSM(SM, Pt, Pt_1, theta)
    % Pt ¡Ù Pt_1 : (Pt .* Pt_1) < 0)
    % Pt_1 > Pt (Selected features have a promoting effect) : (Pt_1 > theta & Pt <= theta)
    SM = SM .* ...
                exp((1 / sqrt(2)) .* ((Pt .* Pt_1) < 0) .* ((Pt_1 > theta & Pt <= theta) - 1/5) );
end