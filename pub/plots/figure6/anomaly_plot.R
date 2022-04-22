library(ggplot2)
library(cowplot)

dnn_extreme_condition$Model = "DNN"
lstm_extreme_condition$Model = "LSTM"
rf_extreme_condition_2$Model = "RF"

df <- rbind(dnn_extreme_condition,rf_extreme_condition_2)
df <- rbind(df, lstm_extreme_condition)

gg1 <- ggplot(df, aes(x=Model, y=lower_quantile)) +
  geom_boxplot() +
  labs(title="Lower Quantile Anomaly Bias",x="Model", y = "Bias")+ theme(plot.title = element_text(face="bold"))


gg2 <- ggplot(df, aes(x=Model, y=normal)) +
  geom_boxplot() +
  labs(title="Normal Conditions Bias",x="Model", y = "Bias")+ theme(plot.title = element_text(face="bold"))


gg3 <- ggplot(df, aes(x=Model, y=upper_quantile)) +
  geom_boxplot() +
  labs(title="Upper Quantile Anomaly Bias",x="Model", y = "Bias")+ theme(plot.title = element_text(face="bold"))

plot_grid( gg1 + geom_hline(yintercept=0, linetype="longdash", color="red")  , gg2 + geom_hline(yintercept=0, linetype="longdash", color="red") , gg3 +  geom_hline(yintercept=0, linetype="longdash", color="red") , labels = c('a', 'b', 'c'), nrow=1)
