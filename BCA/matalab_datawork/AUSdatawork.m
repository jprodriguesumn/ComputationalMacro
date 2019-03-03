% Data loading refers to 1980Q1-2014Q4.
% Deflators for 2012
% Aggregates in current US Dollars

% GDP:  Value, market prices
%PGDP:  Deflator, market prices, Index 2012
%ITISK: Gross capital formation, current prices
%PITISK: Gross capital formation, current prices, deflator, index 2012
% CG: Government final consumption expenditure, value, GDP expenditure approach
%PCG: Government final consumption expenditure, deflator, index 2012
% XGS: Exports of goods and services, value, National Accounts basis
%PXGS: Exports of goods and services, deflator, National Accounts basis
% MGS: Imports of goods and services, value, National Accounts basis
%PMGS: Imports of goods and services, deflator, National Accounts basis
% HRS: Hours worked per employee, total economy
%  ET: Total employment
%  CP: Private final consumption expenditure, value, GDP expenditure approach
% PCP: Private final consumption expenditure, deflator 2012

% set the initial row to read the data from:
startT  = '88'; midsep = ':'; finalT ='227'; % 88 = 1980Q1; 227 = 2014Q4

% Quarterly OECD Economic Outlook flows are originally multiplied by 4
GDP    = xlsread('OECD','AUS',['E',startT,midsep,'E',finalT]);
PGDP   = xlsread('OECD','AUS',['O',startT,midsep,'O',finalT]);
ITISK  = xlsread('OECD','AUS',['G',startT,midsep,'G',finalT]);
PITISK = xlsread('OECD','AUS',['M',startT,midsep,'M',finalT]);
CG     = xlsread('OECD','AUS',['C',startT,midsep,'C',finalT]);
PCG    = xlsread('OECD','AUS',['Q',startT,midsep,'Q',finalT]);
XGS    = xlsread('OECD','AUS',['I',startT,midsep,'I',finalT]);
PXGS   = xlsread('OECD','AUS',['K',startT,midsep,'K',finalT]);
MGS    = xlsread('OECD','AUS',['H',startT,midsep,'H',finalT]);
PMGS   = xlsread('OECD','AUS',['N',startT,midsep,'N',finalT]);
HRS    = xlsread('OECD','AUS',['S',startT,midsep,'S',finalT]);
ET     = xlsread('OECD','AUS',['R',startT,midsep,'R',finalT]);
CP     = xlsread('OECD','AUS',['D',startT,midsep,'D',finalT]);
PCP    = xlsread('OECD','AUS',['P',startT,midsep,'P',finalT]);

% collect residual errors from the aggregate resource constraint
aggcheck = GDP-(CP+ITISK+CG+XGS-MGS);

%get sales taxes
taxdata = xlsread('taxdata',1,'G2:G35'); %sales taxes as % of GDP
%interpolate to fill in missing data - turning yearly data to quarterly
tauc = (interp1(1:size(taxdata,1),taxdata,1:0.25:35+3/4,'linear','extrap'))'/100

% Population data
startT2 = '39'; midsep = ':'; finalT2 ='73';
PoPm  = xlsread('pop.xml',1,['C',startT2,midsep,'D',finalT2]); % population matrix
P     = PoPm(:,2)*10^3; % Population 15-64, persons, thoAUSnds (hence * 10^3).
%interpolate to fill in missing data - turning yearly data to quarterly
iP    = (interp1(1:size(P,1),P,1.25:0.25:size(P,1)+1,'spline','extrap'))' %interpolated to Q


% same deflator for everything
nPGDP = PGDP;
nPITISK = nPGDP; nPCG = nPGDP; nPMGS = nPGDP;
nPXGS = nPGDP; nPCP = nPGDP;

% compute Y, H, X, G: deflate and divide by 4, OECD values are annualized
C = (CP./nPCP./4).*(1-tauc);
Y = (GDP./nPGDP)/4.*(1-tauc);
H = HRS.*ET/4;
X = ITISK./nPITISK./4;
G = CG./nPCG./4 + XGS./nPXGS./4-MGS./nPMGS./4;

% compute per capita values
ypc = Y./iP;
hpc = H./iP/1300;
xpc = X./iP;
gpc = G./iP;
cpc = C./iP;

bdate = 2008.25 %%%make base date beginning of period
bind  = find(t==bdate)
t = (1980.25:0.25:2015)';
mleStartT = 1980.25;
mleFinalT = 2015;
startInd = find(t==mleStartT)
finalInd = find(t==mleFinalT)

global yy by mles mlee
yy   = ypc;
by   = bind;    % base date
mles = startInd; % starting obs index of mle sample
mlee = finalInd; % ending obs index of mle sample

gzt = fsolve(@calgz,0); %solve for growth rate that makes mean detrended log(ypc)= 0 over MLE sample

cpci = ypc-xpc-gpc; % implied consumption as opposed to real consumption cpc
mled  = [t,ypc/ypc(by)*(1+gzt)^by,xpc/ypc(by)*(1+gzt)^by,...
    hpc,gpc/ypc(by)*(1+gzt)^by,cpc/ypc(by)*(1+gzt)^by,cpci/ypc(by)*(1+gzt)^by];

fileid = fopen('..\AUS\ausdata.txt','w')
fprintf(fileid,'%f\n',mled)
%data2 = [ypc xpc hpc gpc cpc iP];