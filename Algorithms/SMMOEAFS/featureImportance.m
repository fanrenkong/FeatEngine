% Calculate feature importance
function value = featureImportance(features, labels)
    %culate MI of a and b in the region of the overlap part

    %�����ص�����
    [Ma,Na] = size(features);
    [Mb,Nb] = size(labels);
    M=min(Ma,Mb);
    N=min(Na,Nb);

    %��ʼ��ֱ��ͼ����
    hab = zeros(256,256);
    ha = zeros(1,256);
    hb = zeros(1,256);

    %��һ��
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

    %ͳ��ֱ��ͼ
    for i=1:M
        for j=1:N
           indexx =  features(i,j);
           indexy = labels(i,j) ;
           hab(indexx,indexy) = hab(indexx,indexy)+1;%����ֱ��ͼ
           ha(indexx) = ha(indexx)+1;%aͼֱ��ͼ
           hb(indexy) = hb(indexy)+1;%bͼֱ��ͼ
       end
    end

    %����������Ϣ��
    hsum = sum(sum(hab));
    index = find(hab~=0);
    p = hab/hsum;
    Hab = sum(sum(-p(index).*log(p(index))));

    %����aͼ��Ϣ��
    hsum = sum(sum(ha));
    index = find(ha~=0);
    p = ha/hsum;
    Ha = sum(sum(-p(index).*log(p(index))));

    %����bͼ��Ϣ��
    hsum = sum(sum(hb));
    index = find(hb~=0);
    p = hb/hsum;
    Hb = sum(sum(-p(index).*log(p(index))));

    %����a��b�Ļ���Ϣ
    value = Ha+Hb-Hab;

    %����a��b�Ĺ�һ������Ϣ
    %mi = hab/(Ha+Hb);
end

