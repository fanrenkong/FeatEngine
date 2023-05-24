function [LeaderSet, LeaderSetFitness] = UpdateLeaderSet(LeaderSet, LeaderSetFitness, N)
    [n, ~] = size(LeaderSet);
    Next = false(1, n);
    [LeaderFrontNo, ~] = NDSort(LeaderSetFitness, 0, n);
    CrowdDis = CrowdingDistance(LeaderSetFitness, LeaderFrontNo);
    [~, Rank] = sort(CrowdDis);
    Next(Rank(1:N)) = true;
    LeaderSet = LeaderSet(Next, :);
    LeaderSetFitness = LeaderSetFitness(Next, :);
end

