% Commonly Used PSO Algorithm (ErFS)
% Programmed by Kevin Kong
function [result, CPUTime] = ErFS(N, Problem, CA, tradeOff)
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
    x = pos > theta;
    fitness = Problem.CalErr(x, CA);
    Problem.nfeval = Problem.nfeval + N; 
    %% Find the optimal
    bestFit = fitness;
    bestPos = pos;
    [GlobalMin, bestIndx] = min(bestFit);
    GlobalMinPos = bestPos(bestIndx, :);
    result.outcome = GlobalMin;
    
    while Problem.NotTerminated()
        % Update the position and velocity of particles
        for i = 1:N
            vel(i, :) = w .* vel(i, :) + c1 .* rand(1, Problem.D) .* (bestPos(i, :) - pos(i, :)) ...
                + c2 .* rand(1, Problem.D) .* (GlobalMinPos - pos(i, :));
            % Fix velocity
            vel(i, :) = ( (vel(i, :) >= Vmin) & (vel(i, :) <= Vmax) ) .* vel(i, :) ...
                + (vel(i, :) < Vmin) .* (Vmin + rand(1, Problem.D) .* (Vmax - Vmin)) ...
                + (vel(i, :) > Vmax) .* (Vmin + rand(1, Problem.D) .* (Vmax - Vmin));
            pos(i, :) = pos(i, :) + vel(i, :);
            % Fix position
            pos(i, :) = ( (pos(i, :) >= Xmin) & (pos(i, :) <= Xmax) ) .* pos(i, :) ...
                + (pos(i, :) < Xmin) .* (Xmin + rand(1, Problem.D) .* (Xmax - Xmin)) ...
                + (pos(i, :) > Xmax) .* (Xmin + rand(1, Problem.D) .* (Xmax - Xmin));
            % Evaluate 
            x = pos(i, :) > theta;
            fitness(i) = Problem.CalErr(x, CA);
            Problem.nfeval = Problem.nfeval + 1;
            % Update the optimal
            if fitness(i) < bestFit(i)
                bestFit(i) = fitness(i);
                bestPos(i, :) = pos(i, :);
                if bestFit(i) < GlobalMin
                    GlobalMin = bestFit(i);
                    GlobalMinPos = bestPos(i, :);
                end
            end
        end
        disp(GlobalMin)
        result.outcome = [result.outcome GlobalMin];
    end
    result.n = 1; % Only one optimal solution
    result.solutions = GlobalMinPos > theta;
    result.objs = GlobalMin;
    CPUTime = toc;
end