clc;clear;
num_samples = 10000; % ����
i = 10;  % ����ź���Ԫ��?
Fs = 128;
N = 128;
windows = 128;
noverlap = 120;

nSamples = N+1;
SNR1 = linspace(10,20,num_samples);
SNR2 = linspace(0,10,num_samples);
SNR3 = linspace(-10,0,num_samples);

g_sum = round(rand(num_samples,i));
f = randi([1,40],num_samples,1);
ask = zeros(num_samples,i*nSamples);
infor = zeros(num_samples,i*nSamples);
%ask_label = 2*ones(num_samples,1);
% Y = zeros(num_samples,N);
% Ayy = zeros(num_samples,N);
seed = [12345 54321];

ask_size = size(ask);
S_size1 = N/2+1;
S_size2 = fix((ask_size(2)-noverlap)/(windows-noverlap));
S = zeros(num_samples,S_size1,S_size2);

for n = 1:num_samples
    [infor(n,:),ask(n,:)] = askd(g_sum(n,:),f(n,:),Fs,N);
    ask(n,:) = hilbert(ask(n,:));
%     randn('state',seed(2));
     ask(n,:) = awgn(ask(n,:),SNR3(n),'measured');
%     Y(n,:) = fft(ask(n,:),N); %��FFT�任
%     Ayy(n,:) = (abs(Y(n,:))); %ȡģ
%     Ayy(n,:) = Ayy(n,:)/(N/2);%�����ʵ�ʵķ��
%     S(n,:,:) = spectrogram(ask(n,:),windows,noverlap,N,Fs);
%     S(n,:,:) = abs(S(n,:,:));
    %figure(1);
%     subplot(221);
%     plot(real(ask(n,:)),'LineWidth',1.5);grid on;
%     axis([0 nSamples*length(g_sum(n,:)) -2.5 2.5]);
%     xlabel('t');
%     ylabel('A');
    [tfr0,t0,f0] = wv(ask(n,:));
%     t = t * 1/1000;
%     [F, T] = meshgrid(f, t);
%     subplot(222);
%     mesh(F, T, abs(tfr));       %����άͼ
%     subplot(223);
    contour(f0,t0,abs(tfr0));
    set(gca,'xtick',[],'xticklabel',[])
    set(gca,'ytick',[],'yticklabel',[])
    saveas(gca,strcat('E:\SNR0\ask\1\',num2str(n),'.jpg'))
end
% for n1 = 1:num_samples
%     [infor(n1,:),ask(n1,:)] = askd(g_sum(n1,:),f(n1,:),Fs,N);
%     ask(n1,:) = hilbert(ask(n1,:));
% 
%      ask(n1,:) = awgn(ask(n1,:),SNR2(n1),'measured');
% 
%     [tfr1,t1,f1] = wv(ask(n1,:));
%     contour(f1,t1,abs(tfr1));
%     set(gca,'xtick',[],'xticklabel',[])
%     set(gca,'ytick',[],'yticklabel',[])
%     saveas(gca,strcat('/home/lab/hzy/wv/ask/1/',num2str(n1+num_samples),'.jpg'))
% end
% for n2 = 1:num_samples
%     [infor(n2,:),ask(n2,:)] = askd(g_sum(n2,:),f(n2,:),Fs,N);
%     ask(n2,:) = hilbert(ask(n2,:));
% 
%      ask(n2,:) = awgn(ask(n2,:),SNR3(n2),'measured');
% 
%     [tfr2,t2,f2] = wv(ask(n2,:));
%     contour(f2,t2,abs(tfr2));
%     set(gca,'xtick',[],'xticklabel',[])
%     set(gca,'ytick',[],'yticklabel',[])
%     saveas(gca,strcat('/home/lab/hzy/wv/ask/1/',num2str(n2+2*num_samples),'.jpg'))
% end
for n3 = 1:num_samples
    im = imread(strcat('E:\SNR0\ask\1\',num2str(n3),'.jpg'));
    im2 = imcrop(im,[115,50,678,536]);
    
    imwrite(im2,strcat('E:\SNR0\ask\2\','ask.',num2str(n3),'.jpg'));
end
% figure;
% % Y = fft(fsk(100,:),N); %��FFT�任
% % Ayy = (abs(Y)); %ȡģ
% plot(Ayy(100,1:N)); %��ʾԭʼ��FFTģֵ���?
% title('FFT ģֵ');
% 
% figure;
% % Ayy=Ayy/(N/2);   %�����ʵ�ʵķ��
% F=([1:N]-1)*Fs/N; %�����ʵ�ʵ�Ƶ���?
% plot(F(1:N/2),Ayy(100,1:N/2));   %��ʾ������FFTģֵ���?
% title('���?Ƶ������ͼ');

% % save ask_samplessnr3 tfr
% % save ask_labels_snr3 ask_label
% figure;
% subplot(2,1,1);plot(infor(100,:),'LineWidth',1.5);grid on;
% title('Binary Signal');
% axis([0 100*length(g_sum(100,:)) -2.5 2.5]);
% 
% subplot(2,1,2);plot(ask(100,:),'LineWidth',1.5);grid on;
% title('FSK modulation');
% axis([0 100*length(g_sum(100,:)) -2.5 2.5]);

