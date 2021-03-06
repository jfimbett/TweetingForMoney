
#************************************************************************************************
#************************************************************************************************
# Name: QL_DA01_familyNames.py
# Author: Juan Felipe Imbett Jiménez
# Date: 23 October 2017 (Monday)
# Description: Stores the names of all fund families used
# 
# Log: 
#
# Version 1: We have 2 list of funds, we merge them and select the unique values. 
#************************************************************************************************

listOfFunds=['BankofHawaii','PalantirTech','PacificFin','Portfolio21Inv','virtus','PIMCO','VoyaInvestments','pacificlife','StadionMoney','ThomasLloyd_','pacificmutual','PinnacleFA','PineBridge','PioneerInvest','PinnacleHolding','PAYDENRYGEL','pensionpartners','PreserverLLC','ProvidentBank','TRowePrice','Prudential','PSAFinancial','PSINext','PekinSinger','PureFunds','PutnamToday','providentira','PZN_News','QuestarCapital','QCIAsset','ohionational','CowenandCompany','Robeco','regalfin','IPOtweet','RiverNorthCap','ReichAndTang','CNRochdale','ParksCapital','RealityShares','RoyceFunds','RWRoge','Ryan_Labs','StandishIM','SegallBryant','SchrodersUS','StatetrustCap','SCMAssociates','snowcapmgt','Stonebridge_Den','SentryInvest','StateFarm','SocieteGenerale','SiriusFunds','SmeadCap','SummitLowVol','Simple_Alts','SNW_Asset_Mgmt','sourceetf','SPAETF','GerrySparrow','SpoutingRockFP','SaratogaCapital','StateStreet','SGAMBOCA','summitportfolio','SWANCapital','synovus','tatrocapital','TocquevilleAM','361Capital','alphatfscap','ThornburgFunds','TIAA','TimothyPlan','TrilliumAM','touchstoneCT','topturncapital','TriadInvestment','TFinancialgrp','Turner_Invest','7TwelveAdv','UBSemea','USCFInvestments','UMBBank','sempercap','USAA','USFunds','UnionStPartners','ttutc','ValueLine','Vontobel_AM','Vanguard_Group','VilasCapital','vaneck_eu','IRandCompliance','WBCapital','VistaPResearch','winchfinancial','WavelengthFunds','WBI_Investments','WesBanco','WestonCapital','WestchesterCap','SalientPartners','WebsterInvstAdv','WeitzInvest','washtrustwealth','WellsFargo','Westfield_Cap','WadeFinancial','WillisGroup','wicfunds','WilshireAdvisor','wintergreenadv','BillHambrecht','WSCBonds','WisdomTreeETFs','WintonCapital','yieldquest','Ziegler_Co','ZevenbergenCap']
otherFunds=['leaseandfinance','AmericanCentury','AmericanFunds',
'AnalyticInvest','AndersonCM_1','AonBenfield','ArmstrongACCA',
'weareartisan','asiahouseuk','AltairAdvisers','alpha_town',
'Thrivent','LoringWard','AstorInvMgmt','aamlive','AbbeyCapital',
'GuideStoneFunds','RaymondJames','AberdeenAssetUS','AcadianAsset',
'AlphaClone','Advance_Capital','Aetna','AngelOakCap','HewittAssociate',
'AIBIreland','InvescoUS','Arrowfunds','AssetMark','ARKInvest','AllstateInv',
'AB_insights','AlsinCapital','Altegris','AllianzGI_view','Amundi_ENG',
'AmBeacon','ArmourWealth','Aspiration','AspenPartners','AspiriantNews'
,'AppletonPtnrs','AQRCapital','arinllc','ArtisansPartner','AscentiaFunds'
,'AshmoreEM','AdvisorShares','AsterCapital','AtlanticTrust','MatsonMoney'
,'grandprixfundF1','avivainvestors','AXAIM','AzzadFunds','BrownAdvisory'
,'rwbaird','BalterLiquidAlt','baronfunds','askBBT','TranswesternPR',
'BainbridgeComp','jpmorgan','BushidoFunds','aboutBGG','Bellinvest',
'BessemerTrust','baxter_intl','BaillieGifford','BrinkerCapital',
'blackrock','WilliamBlairCo','BremerBank','midasfunds','_Finworx',
'BankofOklahoma','BernsteinPWM','Bartlett1898','BTCDM','BernzottCapital',
'CalvertInvests','ColCapMgmt','CallanAssoc','CaritasCapital','CausewayCap',
'CookandBynum','CambiarInvestor','CommerceBank','ClarkCapital',
'Cornerstone_Cap','capitalclough','CallahanAssoc','canecapital',
'CDCgroup','Manta','cliffordcapital','CompassEquity','CastleInfo',
'CGMFunds','CongressAsset','CICompanies','CUNAMutualGroup','CTInvest_US',
'GuggenheimPtnrs','calamos','CLSInvestments','Chase','askcmg','CNBank',
'CowenGroupInc','ComstockPartner','MarketWrap','ccminvests','ColeCapital',
'CreditSuisse','cohenandsteers','CharlesSchwab','CitizensBank',
'PiperJaffrayCo','NoLoadFundX','InvestDavenport','DeutscheBank',
'DividendCapital','dimensional','DayHagan_Invest','Barings',
'DLineCap','DLineFunds','DMSAdvisors','Dodgeandcox','DominiFunds',
'DriehausCapital','Transamerica','DADavidsonCo','MorganStanley',
'EmeraldAssetAdv','ENVintel','EatonVance','EdgeAdvisors',
'EdgemoorInv','EdwardJones','alphaarchitect','EndeavorMgmt',
'EnsembleCapital','EARNESTPartners','EquinoxFunds','3rdAvenueFunds',
'ETFSecuritiesUS','etrade','EvermoreGlobal',
'FirstAmNews','ForumFinancial','FCIAdvisors',
'federatednews','firsteaglefunds','FAMFunds',
'FairfaxGlobal','FHBHawaii','Fidelity','weareforesters',
'fieracapital','fisherinvest','MesirowFin','FirstMerit',
'FocusShares','fortisadvisers','FoxAsset','FortPittCapital'
,'Firstpacific','FundQuest_Adv','FTI_US','FreedomCapital_',
'FrostBank','1stSourceBank','FSBank','ftportfolios',
'FifthThird','ETCETF','FirstTennessee','DanielsTrading',
'MarioGabelli','GAMinsights','GeierFinancial','GersteinFisher'
,'good_harbor','Gateway_LLC','GaveKalCapital','Glenmede',
'GlobalXFunds','AskAIB','GreenwichAdv','GiraldaAdvisors',
'GCAInvestments','GoldmanSachs','hannas55','HeartlandFunds'
,'HavenCapital','HighlandCapMgmt','hilliardlyons','MerrillLynch',
'hartfordfunds','HansonMcClain','HiberniaBank','HancockWhitney',
'HNPCapital','HorizonsETFs','HowardCapMgmt','HarrisOakmark',
'HussmanFunds','hennionandwalsh','HorizonInvest','rbcgamnews'
,'GoIronHorse','NYLandMainStay','IMSCapital','InclineCapital1'
,'INTRUSTBank','leggmason','IronwoodFinance','ISTInsights',
'TheInvestmentH1','ITGinc','JAG_CAPM','JanusCapital',
'JacksonNational','JohnHancockUSA','johnsoninv','Janney1832',
'LincolnInvest','jpmorganfunds','KFWA1','KKR_Co','Kenwood_X','KPMG',
'KaizenAdvisory','LazardAsset','libertyassetmgt','LendLeaseGroup',
'LoganCapitalSD','LongviewCapital','LindeHansenCo','lighthousecapit',
'lordabbett','ManningNapier','meyercapital','MDLFSLtd','MercuryCapAdv',
'followMFS','DeutscheAM_CIO','mercer','Milestone_India','MomentumInvestm','GreatConsumer',
'mai_capital','MckCapital','MarketocracyInc','Everence'
,'montereychat','MainManagement','massmutual','manarinonmoney',
'MandT_Bank','jeffnat','InvestCIP','marshfielding','Mosaic_Cap_Corp','MTBIA','InvestMontage','MatherandCo_ltd','MunderCapital',
'MeyerCapital1',
'AllianzGI_US','Navellier','NiemannCapital','moneymanagement',
'neubergerberman','_NEIRG_','thinknewfound','Nationwide',
'NorthpointCap','NorthernTrust','NuanceInvest','ManGroup',
'NuveenInv','NatixisGlobalAM','NM_News','NewYorkLife',
'Oaktree','OppFunds','PaxWorld', 'BankofHawaii','PalantirTech','PacificFin',
'Portfolio21Inv','virtus','PIMCO','VoyaInvestments','pacificlife','StadionMoney',
'ThomasLloyd_','pacificmutual','PinnacleFA','PineBridge','PioneerInvest',
'PinnacleHolding','PAYDENRYGEL','pensionpartners','PreserverLLC',
'ProvidentBank','TRowePrice','Prudential','PSAFinancial',
'PSINext','PekinSinger','PureFunds','PutnamToday',
'providentira','PZN_News','QuestarCapital','QCIAsset',
'ohionational','CowenandCompany','Robeco','regalfin','IPOtweet','RiverNorthCap','ReichAndTang',
'CNRochdale','ParksCapital','RealityShares','RoyceFunds','RWRoge','Ryan_Labs','StandishIM',
'SegallBryant','SchrodersUS','StatetrustCap','SCMAssociates','snowcapmgt','Stonebridge_Den'
,'SentryInvest','StateFarm','SocieteGenerale','SiriusFunds','SmeadCap','SummitLowVol',
'Simple_Alts','SNW_Asset_Mgmt','sourceetf','SPAETF','GerrySparrow','SpoutingRockFP',
'SaratogaCapital','StateStreet','SGAMBOCA','summitportfolio','SWANCapital',
'synovus','tatrocapital','TocquevilleAM','361Capital','alphatfscap','ThornburgFunds','TIAA','TimothyPlan','TrilliumAM','touchstoneCT',
'topturncapital','TriadInvestment',
'TFinancialgrp','Turner_Invest',
'7TwelveAdv','UBSemea','USCFInvestments','UMBBank','sempercap',
'USAA','USFunds','UnionStPartners','ttutc','ValueLine','Vontobel_AM',
'Vanguard_Group','VilasCapital','vaneck_eu','IRandCompliance','WBCapital',
'VistaPResearch','winchfinancial','WavelengthFunds','WBI_Investments',
'WesBanco','WestonCapital','WestchesterCap','SalientPartners','WebsterInvstAdv',
'WeitzInvest','washtrustwealth','WellsFargo','Westfield_Cap','WadeFinancial',
'WillisGroup','wicfunds','WilshireAdvisor','wintergreenadv','BillHambrecht',
'WSCBonds','WisdomTreeETFs','WintonCapital','yieldquest','Ziegler_Co','ZevenbergenCap']

external=[ 'SeekingAlpha', 'MarketCurrents', 'business', 'markets','MorningstarInc','FT', 'WSJ', 'CityFalcon', 'smallcappower', 'FinancialNews', 'MstarAdvisor']
last=[
"WilliamBlairIM",
"TouchstoneAdvCT",
"AmundiPioneer",
"JPMorganAM",
"ColonyGroup",
"CalvertUpdates",
"NatixisIM",
"WSC_Bonds",
"WBIInvestments",
"UBS",
"CallanLLC ",
"InvescoUS",
"PekinHardy",
"Aon_plc",
"CapitalGroup",
"NYLInvestments",
"advisorhubinc",
"DWS_Group",
"NAS_Nationwide",
"ISTNetworks",
"CIMinfo_Finance",
"GreentechCap",
"joulefinancial"
]
