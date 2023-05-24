function [Archive, ArchiveFitness] = UpdateArchive(Archive, ArchiveFitness, Offspring, OffspringFitness, epsilon)
    [N, ~] = size(Archive);
    %% Calculate the grid location of each solution
    ChildGrid = floor((OffspringFitness-min(ArchiveFitness, [], 1))/epsilon);
    ArchiveGrid = floor(ArchiveFitness-repmat(min(ArchiveFitness, [], 1), N, 1)/epsilon);
    
    %% Insert the offspring into the archive by epsilon-dominance and grid locations
    if ~any(all(ArchiveGrid<=ChildGrid, 2))
        Dominate = find(all(ChildGrid <= ArchiveGrid, 2));
        if ~isempty(Dominate)
            Archive(Dominate, :) = [];
            ArchiveFitness(Dominate, :) = [];
            Archive = [Archive; Offspring];
            ArchiveFitness = [ArchiveFitness; OffspringFitness];
        else
            SameGrid = find(ismember(ArchiveGrid, ChildGrid, 'rows'), 1);
            if isempty(SameGrid)
                Archive = [Archive; Offspring];
                ArchiveFitness = [ArchiveFitness; OffspringFitness];
            else
                B = ChildGrid * epsilon + min(ArchiveFitness, [], 1);
                if norm(OffspringFitness - B) < norm(ArchiveFitness(SameGrid, :) - B)
                    Archive(SameGrid, :) = Offspring;
                    ArchiveFitness(SameGrid, :) = OffspringFitness;
                end
            end
        end
    end
end