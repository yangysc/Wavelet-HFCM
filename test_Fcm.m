clc
ctime = datestr(now, 30);

tseed = str2num(ctime((end - 5) : end)) ;

rand('seed', tseed) ;
enrollment = [13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919, 16388, 15433, 15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328, 19337, 18876]';
maxV = max(enrollment);
minV = min(enrollment);
normalizeData = (enrollment - minV)./ (maxV - minV);
[center, U, obj_func] = fcm(normalizeData, 3);
for i = 1:3
    plot(U(i, :), '-*')
    hold on
end
fprintf('聚类中心为：\n');
disp(center)
