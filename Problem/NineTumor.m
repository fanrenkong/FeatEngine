classdef NineTumor < PROBLEM
    % 9 Tumor Data Set
    
    methods
        function obj = NineTumor(nfevalmax)
            obj@PROBLEM(nfevalmax);
            load('Datasets/9Tumor.mat');
            obj.all_features = features;
            obj.labels = labels;
            [~, obj.D] = size(obj.all_features);
            testRatio = 0.3;
            obj = obj.trainTestSplit(testRatio);
            clearvars features labels
        end
    end
end

