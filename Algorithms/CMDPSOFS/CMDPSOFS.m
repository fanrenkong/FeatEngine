% PSO-Based Multi-Objective Feature Selection Algorithm 2
% OMOPSO
% Programmed by Kevin Kong
function [result, CPUTime] = CMDPSOFS(N, Problem, CA, tradeOff)
    tic;
    Problem.nfeval = 0;
    % Threshold used to judge if a feature is selected in real encoding
    theta = 0;
    % Lower bound of decision variable
    lb = -ones(1, Problem.D); 
    % Upper bound of decision variable
    ub = ones(1, Problem.D);
    % PSO parameters
    Xmax = ub;
    Xmin = lb;
    Vmax = 0.5 * (Xmax - Xmin);
    Vmin = -Vmax;
    % Mutation parameters
    muProb = 1 / Problem.D; % mutation probability
    % Dominance parameters
    epsilon = N;
    %% Initialize position and velocity of particles
    pos = realInit(N, Xmin, Xmax);
    vel = realInit(N, Vmin, Vmax);
    %% Initialize LeaderSet
    LeaderSet = realInit(N, Xmin, Xmax);
    %% Initialize Archive
    Archive = LeaderSet;
    %% Evaluate
    fitness = zeros(N, 2);
    LeaderSetFitness = zeros(N, 2);
    x = pos > theta;
    leaderX = LeaderSet > theta;
    fitness(:, 1) = Problem.CalErr(x, CA);
    fitness(:, 2) = Problem.CalFeatsCount(x, true);
    LeaderSetFitness(:, 1) = Problem.CalErr(leaderX, CA);
    LeaderSetFitness(:, 2) = Problem.CalFeatsCount(x, true);
    ArchiveFitness = LeaderSetFitness;
    Problem.nfeval = Problem.nfeval + 2*N; 
    %% Calculate the crowding distance of LeaderSet
    [LeaderSetFrontNo, ~] = NDSort(LeaderSetFitness, zeros(N, 1), N);
    LeaderSetCrowdDis = CrowdingDistance(LeaderSetFitness, LeaderSetFrontNo);
    %% Find pbest
    pbestFit = fitness;
    pbestPos = pos;
    while Problem.NotTerminated()
        ImprovePos = [];
        ImproveFit = [];
        for i = 1:N
            % randomly select a gbest
            gbestIndex = TournamentSelection(2, 1, LeaderSetCrowdDis);
            % PSO parameters
            w = 0.1 + rand * 0.4;
            c1 = 1.5 + rand * 0.5;
            c2 = 1.5 + rand * 0.5;
            % update
            vel(i, :) = w .* vel(i, :) + c1 .* rand(1, Problem.D) .* (pbestPos(i, :) - pos(i, :)) ...
                + c2 .* rand(1, Problem.D) .* (pos(gbestIndex, :) - pos(i, :));
            % Fix velocity
            vel(i, :) = ( (vel(i, :) >= Vmin) & (vel(i, :) <= Vmax) ) .* vel(i, :) ...
                + (vel(i, :) < Vmin) .* (Vmin + rand(1, Problem.D) .* (Vmax - Vmin)) ...
                + (vel(i, :) > Vmax) .* (Vmin + rand(1, Problem.D) .* (Vmax - Vmin));
            pos(i, :) = pos(i, :) + vel(i, :);
            % Mutation
            group = mod(i, 3);
            if group == 0
                % uniform mutation
                pos(i, :) = UniformMutation(pos(i, :), muProb);
            elseif group == 1
                % nonuniform mutation
                t = Problem.nfeval / Problem.nfevalmax;
                pos(i, :) = NonUniformMutation(pos(i, :), muProb, t);
            end
            
            % Fix
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
                ImprovePos = [ImprovePos; pos(i, :)];
                ImproveFit = [ImproveFit; fitness(i, :)];
            end
        end
        % Update LeaderSet
        LeaderSet = [LeaderSet; ImprovePos];
        LeaderSetFitness = [LeaderSetFitness; ImproveFit];
        [LeaderSet, LeaderSetFitness] = UpdateLeaderSet(LeaderSet, LeaderSetFitness, N);
        % Update Archive
        [Archive, ArchiveFitness] = UpdateArchive(Archive, ArchiveFitness, pos, fitness, epsilon);
        % Calculate the crowding distance of LeaderSet
        [LeaderSetFrontNo, ~] = NDSort(LeaderSetFitness, zeros(N, 1), N);
        LeaderSetCrowdDis = CrowdingDistance(LeaderSetFitness, LeaderSetFrontNo);
    end
    CPUTime = toc;
    [FrontNo, ~] = NDSort(fitness, zeros(N, 1), N);
    F1 = FrontNo==1;
    result.n = sum(F1);
    result.solutions = pos(F1, :) > theta;
    result.objs = fitness(F1, :);
end