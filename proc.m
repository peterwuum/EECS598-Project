evaluation = dir('*evaluation');
fname = extractfield(evaluation,'name');
fname = transpose(fname)

lam_seq = 0.1:0.05:0.65;
gamma_seq = 0.05:0.05:0.25;
train_acc = zeros(length(lam_seq),length(gamma_seq));
test_acc = zeros(length(lam_seq),length(gamma_seq));

for i = 1:length(lam_seq)
    for j = 1:length(gamma_seq)
        filename = fname{((i-1)*length(gamma_seq)+j)};
        res = importdata('./OUTPUT/' + filename);
        res_num = res.data;
        train_acc(i,j) = res_num(1);
        test_acc(i,j) = res_num(2);
    end
end

f1 = figure(1);
colormap('hot')
imagesc(train_acc)
colorbar;
set(gca,'XTick', 1:1:5)
set(gca,'YTick',1:1:12)
axis on; 
ax1 = f1.CurrentAxes
Lx1=ax1.XTickLabel
Lx2=linspace(0.05,0.25,numel(Lx1))
Lx_cell={};
for k=1:1:numel(Lx1)
    L1=num2str(Lx2(k));
    Lx_cell=[Lx_cell L1];
end
Lx_cell=Lx_cell';

f1.CurrentAxes.XTickLabel=Lx_cell
Ly1=ax1.YTickLabel
Ly2=linspace(0.10,0.65,numel(Ly1))
Ly_cell={};
for k=1:1:numel(Ly1)
    L2=num2str(Ly2(k));
    Ly_cell=[Ly_cell L2];
end
Ly_cell=Ly_cell';

f1.CurrentAxes.YTickLabel=Ly_cell
xlabel('gamma')
ylabel('lambda')


f2 = figure(2);
colormap('hot')
imagesc(test_acc)
colorbar;
set(gca,'XTick', 1:1:5)
set(gca,'YTick',1:1:12)
axis on; 
ax1 = f2.CurrentAxes
Lx1=ax1.XTickLabel
Lx2=linspace(0.05,0.25,numel(Lx1))
Lx_cell={};
for k=1:1:numel(Lx1)
    L1=num2str(Lx2(k));
    Lx_cell=[Lx_cell L1];
end
Lx_cell=Lx_cell';

f2.CurrentAxes.XTickLabel=Lx_cell
Ly1=ax1.YTickLabel
Ly2=linspace(0.10,0.65,numel(Ly1))
Ly_cell={};
for k=1:1:numel(Ly1)
    L2=num2str(Ly2(k));
    Ly_cell=[Ly_cell L2];
end
Ly_cell=Ly_cell';

f2.CurrentAxes.YTickLabel=Ly_cell
xlabel('gamma')
ylabel('lambda')


