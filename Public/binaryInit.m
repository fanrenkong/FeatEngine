function PopDec = binaryInit(N, D)
    % Initialize population by binary encoding
    PopDec = randi([0, 1], N, D); 
end