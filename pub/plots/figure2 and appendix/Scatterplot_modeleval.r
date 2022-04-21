library(readr)
library(ggrepel)
library(plyr)
library(cowplot)

#importing df
rf_R2_n200_d10 <- read_csv("rf_R2_n200_d10.csv")
rf_R2_n200_d10<-rf_R2_n200_d10[,2:3]
rf[,1]<-rf_R2_n200_d10[,2]
rf[,2]<-rf_R2_n200_d10[,1]

fcn<- read_csv("fcn_notime_predictions.csv")
lstm<- read_csv("lstm_r2_sites.csv")
phy<- read_csv("df_metrics_oob_stocker20gmd_fig2.csv")
lstm_cond<- read_csv("lstm_conditional_r2_sites.csv")
colnames(lstm_cond)[1]<-'site'
colnames(lstm_cond)[2]<-'lstm_cond_R2'

colnames(lstm)[1]<-"site"
colnames(lstm)[2]<-"lstm_R2"
fcn<-fcn[,2:3]

colnames(fcn)[1]<-"site"
colnames(fcn)[2]<-"fcn_R2"
rf<-rf[,1:2]
colnames(rf)[1]<-"site"
colnames(rf)[2]<-"rf_R2"
phy<-phy[,c("site","rsq")]
colnames(phy)[2]<-"phy_R2"

rf_lstm<-merge(rf,lstm,by="site")
and_phy<-merge(rf_lstm,phy,by="site")
and_fcn<-merge(fcn,and_phy,by="site")      
df<- merge(and_fcn,lstm_cond,by="site")      
df[which(df[,"rf_R2"]<0),][,"rf_R2"]<-0


#plotting  

#LSTM vs RF
iind<-which(df$rf_R2<0.6|df$rf_R2>df$lstm_R2)
naame<-df$site[iind]
df_lstm_rf<-df
df_lstm_rf$site<-""
df_lstm_rf[iind,]$site<-naame

p<-ggplot(df_lstm_rf, aes(rf_R2,lstm_R2, label =site)) +
  geom_point(color = "red") + labs(title = "Leave-site-out cross-validation (Random Forest)")+
  geom_text_repel(min.segment.length =0,box.padding = 0.5)+geom_abline(intercept=0, slope =1, color="black",linetype = "dashed")+xlab(expression("Rondom forest R"^2)) + ylab(expression("LSTM R"^2))+
  xlim(c(0,1))+ylim(c(0,1))+theme(plot.title = element_text(face = "bold"))
p
#ggsave("Scatter_LSTM_RF.png")

#LSTM vs FCN
ind2<-which(df$fcn_R2<0.6|df$fcn_R2>df$lstm_R2)
name2<-df$site[ind2]
df_lstm_fcn<-df
df_lstm_fcn$site<-""
df_lstm_fcn[ind2,]$site<-name2
p1<-ggplot(df_lstm_fcn, aes(fcn_R2,lstm_R2, label =site)) +
  geom_point(color = "red") + labs(title = "Leave-site-out cross-validation (Fully-connected NN)")+
  geom_text_repel(min.segment.length =0,box.padding = 0.5)+geom_abline(intercept=0, slope =1,  color="black",linetype = "dashed")+xlab(expression("Fully-connected Neural Network R"^2)) + ylab(expression("LSTM R"^2))+
  xlim(c(0,1))+ylim(c(0,1))+theme(plot.title = element_text(face = "bold"))
p1

#ggsave("Scatter_LSTM_fcn.pdf")

# put plots together for report
plot_grid(p1,p, labels = c('b','c'))


# LSTM and Physical
ind<-which(df$phy_R<0.6|df$phy_R>df$lstm_R2)
name<-df$site[ind]
df_lstm_phy<-df
df_lstm_phy$site<-""
df_lstm_phy[ind,]$site<-name
p2<-ggplot(df_lstm_phy, aes(phy_R2,lstm_R2, label = site)) +
  geom_point(color = "red") + labs(title = "Leave-site-out cross-validation (Physical Model)")+
  geom_text_repel(min.segment.length =0,box.padding = 0.7)+geom_abline(intercept=0, slope =1,  color="black",linetype = "dashed")+xlab(expression("Physical Model R"^2)) + ylab(expression("LSTM R"^2))+
  xlim(c(0,1))+ylim(c(0,1))+theme(plot.title = element_text(face = "bold"))
p2
plot_grid(p2, labels = c('a'))
#ggsave("Scatter_LSTM_Physical.pdf")

# LSTM and LSTM condition
ind1<-which(df$lstm_cond_R2<0.7)
name1<-df$site[ind1]
df_lstm_lstmcond<-df
df_lstm_lstmcond$site<-""
df_lstm_lstmcond[ind1,]$site<-name1

p3<-ggplot(df_lstm_lstmcond, aes(lstm_cond_R2,lstm_R2, label =site))+
  geom_point(color = "red") + labs(title = "Leave-site-out cross-validation (LSTM with Condition)")+
  geom_text_repel(min.segment.length =0,seed = 42,box.padding = 0.5)+geom_abline(intercept=0, slope =1,  color="black",linetype = "dashed")+xlab(expression("LSTM with Condition R"^2)) + ylab(expression("LSTM R"^2))+
  xlim(c(0,1))+ylim(c(0,1))+theme(plot.title = element_text(face = "bold"))
p3
#ggsave("Scatter_LSTM_LSTMCOND.pdf")



