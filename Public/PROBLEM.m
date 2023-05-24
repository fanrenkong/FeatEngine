classdef PROBLEM
    % PROBLEM - The superclass of all the problems.
    properties
        all_features; 
        labels;
        D;
        nfevalmax;
        nfeval;
        trainIdx;
        testIdx;
    end
    
    methods
        %% Constructor
        function obj = PROBLEM(nfevalmax)
            obj.nfevalmax = nfevalmax;
            obj.nfeval = 0;
        end
        
        %% Split dataset into train and test subsets
        function obj = trainTestSplit(obj, testRatio)
            % Create the object of layered random division
            c = cvpartition(obj.labels, 'HoldOut', testRatio);
            % Get train set index
            obj.trainIdx = training(c);
            % Get test set index
            obj.testIdx = test(c);
        end
        
        %% Determine if terminated
        function bool = NotTerminated(obj)
            bool = (obj.nfeval < obj.nfevalmax);
            fprintf('progress: (%2d/%2d)\n', obj.nfeval, obj.nfevalmax);
        end
        %% Calculate loss value
        function loss = CalLoss(obj, PopDec, func)
            [N, ~] = size(PopDec);
            loss = zeros(N, 1);
            n_samples = 60;
            for i = 1 : N
                % Get selected features in X_i
                x = obj.all_features(1:n_samples, PopDec(i, :) == 1);
                y = obj.labels(1:n_samples, :);
                % Train model
                Model = func(x, y);
                % Cross valid
                CVModel = crossval(Model);
                % Calculate loss rate
                loss(i) = kfoldLoss(CVModel);
                
            end
        end
        
        %% Calculate error rate
        function err = CalErr(obj, PopDec, func)
            [N, ~] = size(PopDec);
            err = zeros(N, 1);
            for i = 1 : N
                % Get selected features in X_i
                trainFeats = obj.all_features(obj.trainIdx, PopDec(i, :) == 1);
                trainLabels = obj.labels(obj.trainIdx, :);
                testFeats = obj.all_features(obj.testIdx, PopDec(i, :) == 1);
                testLabels = obj.labels(obj.testIdx, :);
                % Train model
                Model = func(trainFeats, trainLabels);
                % Predict
                predictions = predict(Model, testFeats);
                % Calculate consistency between predicted results and real
                % labels
                correctPredictions = (predictions == testLabels);
                % True positive
                TP = sum(correctPredictions & predictions);
                % True negative
                TN = sum(correctPredictions & ~predictions);
                % False positive
                FP = sum(~correctPredictions & predictions);
                % False negative
                FN = sum(~correctPredictions & ~predictions);
                err(i) = (FP + FN) / (TP + TN + FP + FN);
            end
        end
        
        %% Count features 
        function count = CalFeatsCount(obj, PopDec, normal)
            count = sum(PopDec, 2);
            if normal
                count = count ./ obj.D;
            end
        end
    end
end

