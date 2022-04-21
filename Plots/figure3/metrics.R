library(ggplot2)
library(latex2exp)
library(cowplot)

r2_lstm = data.frame(r2 = lstm_metrics$R2)
r2_dnn = data.frame(r2 = metrics_dnn$r2)

r2_lstm$Model <- "LSTM"
r2_dnn$Model <- "DNN"

r2 <- rbind(r2_lstm, r2_dnn)

p1 <- ggplot(r2, aes(r2, fill = Model)) + 
  geom_histogram(alpha = 0.6
                 ,aes(y=..density..),binwidth=0.1, position = 'identity') + xlab(expression(R^2)) + ylab("Density") + ggtitle(TeX("\\textbf{Out-of-sample $R^2$} "))


rmse_lstm = data.frame(rmse = lstm_metrics$RMSE)
rmse_dnn = data.frame(rmse = metrics_dnn$rmse)

rmse_lstm$Model <- "LSTM"
rmse_dnn$Model <- "DNN"

rmse <- rbind(rmse_lstm, rmse_dnn)

p2 <- ggplot(rmse, aes(rmse, fill = Model)) + 
  geom_histogram(alpha = 0.6
                 ,aes(y=..density..),binwidth=0.075, position = 'identity') + xlab(TeX("RMSE $(g \\; C \\; m^{-2} \\; d^{-1})$")) + ylab("Density") + ggtitle(TeX("\\textbf{Out-of-sample RMSE} "))

plot_grid(p1   , p2  , labels = c('a', 'b'), nrow=1)
