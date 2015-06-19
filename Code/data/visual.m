
load('06-ADNI.mat');
data = dataAll{1};

nSmp = size(data.X,2);
[data.RID, idx] = sort(data.RID,'ascend');
RID = unique(data.RID);
data.month = data.month(idx);
data.gndDX = data.gndDX(idx);
v = [data.RID', data.month', data.gndDX'];

figure; title('X-month, Y-subject');
for i = 1:nSmp
    y = find(RID==data.RID(i));
    x = data.month(i);
    switch data.gndDX(i)
        case 0
            plot(x,y,'go','LineWidth',2); hold on;
        case 1
            plot(x,y,'o','LineWidth',2, 'Color', [1,0.7,0]); hold on;
        case 2
            plot(x,y,'ro','LineWidth',2); hold on;
    end
end

figure; title('X-month, Y-subject');
for set = 1:9
    subplot(3,3,set);
    for i = (set-1)*22+1:min(set*22,nSmp)
        y = find(RID==data.RID(i));
        x = data.month(i);
        switch data.gndDX(i)
            case 0
                plot(x,y,'go','LineWidth',2); hold on;
            case 1
                plot(x,y,'o','LineWidth',2, 'Color', [1,0.7,0]); hold on;
            case 2
                plot(x,y,'ro','LineWidth',2); hold on;
        end
    end
end

