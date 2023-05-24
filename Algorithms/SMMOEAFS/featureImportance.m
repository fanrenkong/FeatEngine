% Calculate feature importance
function value = featureImportance(features, labels)
    %culate MI of a and b in the region of the overlap part

    %计算重叠部分
    [Ma,Na] = size(features);
    [Mb,Nb] = size(labels);
    M=min(Ma,Mb);
    N=min(Na,Nb);

    %初始化直方图数组
    hab = zeros(256,256);
    ha = zeros(1,256);
    hb = zeros(1,256);

    %归一化
    if max(max(features))~=min(min(features))
        features = (features-min(min(features)))/(max(max(features))-min(min(features)));
    else
        features = zeros(M,N);
    end

    if max(max(labels))-min(min(labels))
        labels = (labels-min(min(labels)))/(max(max(labels))-min(min(labels)));
    else
        labels = zeros(M,N);
    end

    features = double(int16(features*255))+1;
    labels = double(int16(labels*255))+1;

    %统计直方图
    for i=1:M
        for j=1:N
           indexx =  features(i,j);
           indexy = labels(i,j) ;
           hab(indexx,indexy) = hab(indexx,indexy)+1;%联合直方图
           ha(indexx) = ha(indexx)+1;%a图直方图
           hb(indexy) = hb(indexy)+1;%b图直方图
       end
    end

    %计算联合信息熵
    hsum = sum(sum(hab));
    index = find(hab~=0);
    p = hab/hsum;
    Hab = sum(sum(-p(index).*log(p(index))));

    %计算a图信息熵
    hsum = sum(sum(ha));
    index = find(ha~=0);
    p = ha/hsum;
    Ha = sum(sum(-p(index).*log(p(index))));

    %计算b图信息熵
    hsum = sum(sum(hb));
    index = find(hb~=0);
    p = hb/hsum;
    Hb = sum(sum(-p(index).*log(p(index))));

    %计算a和b的互信息
    value = Ha+Hb-Hab;

    %计算a和b的归一化互信息
    %mi = hab/(Ha+Hb);
end

