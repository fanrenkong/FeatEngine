% PSO-Based Multi-Objective Feature Selection Algorithm 1
% Programmed by Kevin Kong
function [result, CPUTime] = NSPSOFS(N, Problem, CA, tradeOff)
    tic;
    Problem.nfeval = 0;
    % Threshold used to judge if a feature is selected in real encoding
    theta = 0;
    % Lower bound of decision variable
    lb = -ones(1, Problem.D); 
    % Upper bound of decision variable
    ub = ones(1, Problem.D);
    % PSO parameters
    w = 0.72984;
    c1 = 2;
    c2 = 2;
    Xmax = ub;
    Xmin = lb;
    Vmax = 0.5 * (Xmax - Xmin);
    Vmin = -Vmax;
    %% Initialize position and velocity of particles
    pos = realInit(N, Xmin, Xmax);
    vel = realInit(N, Vmin, Vmax);
    %% Evaluate
    fitness = zeros(N, 2);
    x = pos > theta;
    fitness(:, 1) = Problem.CalErr(x, CA);
    fitness(:, 2) = Problem.CalFeatsCount(x, true);
    Problem.nfeval = Problem.nfeval + N; 
    %% Find pbest
    pbestFit = fitness;
    pbestPos = pos;
    while Problem.NotTerminated()
        % Identify the non-dominated solutions
        [FrontNo, ~] = NDSort(fitness, zeros(N,1), N);
        CrowdDis = CrowdingDistance(fitness, FrontNo);
        % Copy all pariticles to a union
        unionPos = pos;
        unionFitness = fitness;
        for i = 1:N
            % randomly select a gbest
            gbestIndex = selectGbest(CrowdDis, FrontNo);
            vel(i, :) = w .* vel(i, :) + c1 .* rand(1, Problem.D) .* (pbestPos(i, :) - pos(i, :)) ...
                + c2 .* rand(1, Problem.D) .* (pos(gbestIndex, :) - pos(i, :));
            % Fix velocity
            vel(i, :) = ( (vel(i, :) >= Vmin) & (vel(i, :) <= Vmax) ) .* vel(i, :) ...
                + (vel(i, :) < Vmin) .* (Vmin + rand(1, Problem.D) .* (Vmax - Vmin)) ...
                + (vel(i, :) > Vmax) .* (Vmin + rand(1, Problem.D) .* (Vmax - Vmin));
            pos(i, :) = pos(i, :) + vel(i, :);
            pos(i, :) = ( (pos(i, :) >= Xmin) & (pos(i, :) <= Xmax) ) .* pos(i, :) ...
                + (pos(i, :) < Xmin) .* (Xmin + rand(1, Problem.D) .* (Xmax - Xmin)) ...
                + (pos(i, :) > Xmax) .* (Xmin + rand(1, Problem.D) .* (Xmax - Xmin));
            % Evaluate
            x = pos(i, :) > theta;
            fitness(i, 1) = Problem.CalErr(x, CA);
            fitness(i, 2) = Problem.CalFeatsCount(x, true);
            Problem.nfeval = Problem.nfeval + 1;
            % Update the pbest
            % if new position dominates the current pbest(i)
            [FrontNoPbest, ~] = NDSort([fitness(i, :);pbestFit(i, :)], zeros(2, 1), 2);
            if FrontNoPbest(1) < FrontNoPbest(2)
                % better
                pbestFit(i, :) = fitness(i, :);
                pbestPos(i, :) = pos(i, :);
            end
        end
        unionPos = [unionPos; pos];
        unionFitness = [unionFitness; fitness];
        %% Get next generation population
        % Non-dominated sorting
        [FrontNoUnion, MaxFNoUnion] = NDSort(unionFitness, zeros(2*N, 1), 2*N);
        Next = false(1, 2*N);
        % Select the solutions in the last front based on their crowding
        i = 1;
        while sum(Next) < N && i <= MaxFNoUnion
            Fi = find(FrontNoUnion == i);
            if sum(Next) + length(Fi) <= N
                % add Fi to swarm
                Next(Fi) = true;
                i = i + 1;
            else
                % exceed the capacity
                % calculate the crowding distance of each solution
                CrowdDis = CrowdingDistance(unionFitness, FrontNoUnion);
                [~, Rank] = sort(CrowdDis(Fi), 'descend');
                Next(Fi(Rank(1:N-sum(Next)))) = true;
            end           
        end
        % Population for next generation
        pos = unionPos(Next, :);
        fitness = unionFitness(Next, :);
    end
    CPUTime = toc;
    [FrontNo, ~] = NDSort(fitness, zeros(N, 1), N);
    F1 = FrontNo==1;
    result.n = sum(F1);
    result.solutions = pos(F1, :) > theta;
    result.objs = fitness(F1, :);
end
%% Randomly select a gbest
function gbestIndex = selectGbest(CrowdDis, FrontNo)
    nonDomCrowdDis = CrowdDis(FrontNo == 1);
    nonDomCrowdDis = sort(nonDomCrowdDis);
    leastCrowdedValue = nonDomCrowdDis(1);
    leastCrowdedIndex = find(CrowdDis == leastCrowdedValue & FrontNo == 1);
    gbestIndex = leastCrowdedIndex(randi(numel(leastCrowdedIndex)));
end