# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:31:45 2019

@author: joph
"""



from importlib import reload

#import ml_classes


#https://en.wikipedia.org/wiki/Universal_approximation_theorem
#https://tryolabs.com/blog/2018/12/19/major-advancements-deep-learning-2018/
#https://www.sciencedirect.com/science/article/pii/S2542435118305701

x = np.random.random(10000)*10
y1 = x 
y2 = x*x
y3 = np.sin(x)

lags=100

lagged=np.zeros((len(x)-lags,lags))

for i in range(0,len(x)-lags):
    lagged[i,:]=x[i:i+lags]
    

model1= sm.OLS(y1, x).fit()
model2= sm.OLS(y2, x).fit()
model3= sm.OLS(y3, x).fit()
model4= sm.OLS(y3[0:(len(x)-lags)], lagged).fit()


############model1
fig = sm.graphics.plot_fit(model1,0)

fig = sm.graphics.plot_fit(model2,0)

fig = sm.graphics.plot_fit(model3,0)

fig= sm.graphics.plot_fit(model4,0)


model = Sequential()
model.add(Dense(20 , activation='tanh',input_dim=1))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(20,activation='tanh'))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1,activation='tanh'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

def linear_f(x):
    return x

def quadratic_f(x):
    return x*x

def sin_f(x):
    return np.sin(x)

n=5000
batch_size=1
max_elem=2500

#linear=Function_data_no_ts(n,10,max_elem,model,linear_f,1)
#linear.fit_model_plot_results(epochs,batch_size)

#linear=Function_data_no_ts(n,10,max_elem,model,quadratic_f,1)
#linear.fit_model_plot_results(epochs,batch_size)

#linear=Function_data_no_ts(n,10,max_elem,model,sin_f,1)
#linear.fit_model_plot_results(epochs,batch_size)

lags=1
epochs=5

#reload(ml_classes)


model = Sequential()
model.add(Dense(2,input_dim=lags))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])


linear=Function_data_no_ts(n,10,max_elem,model,linear_f,lags)
linear.fit_model_plot_results(epochs,batch_size)

model = Sequential()
model.add(Dense(5, activation='tanh',input_dim=lags))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(10,activation='tanh'))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1,activation='tanh'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

linear=Function_data_no_ts(n,10,max_elem,model,quadratic_f,lags)
linear.fit_model_plot_results(epochs,batch_size)

model = Sequential()
model.add(Dense(80, activation='tanh',input_dim=lags))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(40,activation='tanh'))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1,activation='tanh'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

linear=Function_data_no_ts(n,10,max_elem,model,sin_f,lags)
linear.fit_model_plot_results(epochs,batch_size)

model = Sequential()
model.add(LSTM(20, batch_input_shape=(1, lags, 1), return_sequences=True, stateful=True))
model.add(LSTM(20, return_sequences=False, stateful=True))
model.add(Dense(20))
model.add(Dense(1))
model.compile(loss='mse', optimizer=adam(lr=0.0001))

linear=Function_data_lstm(n,10,max_elem,model,linear_f,lags)
linear.fit_model_plot_results(epochs,batch_size)

quadratic=Function_data_lstm(n,10,max_elem,model,quadratic_f,lags)
quadratic.fit_model_plot_results(epochs,batch_size)

sine=Function_data_lstm(n,10,max_elem,model,sin_f,lags)
sine.fit_model_plot_results(epochs,batch_size)





#model = Sequential()
#model.add(LSTM(4, input_shape=(1, look_back)))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
#testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
#plt.scatter(x,y1)
plt.scatter(x,y3)
plt.plot(x,y3)


