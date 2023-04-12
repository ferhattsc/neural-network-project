import csv
import numpy as np

class Network: 
    
    # Agirliklar ve bias
    def __init__(self):
        # 6 adet agirlik ve 3 adet bias degeri
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
      
    def sigmoid(self , x): 
        #f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_turev(self , x):
        #f'(x) = f(x) * (1 - f(x))
        sig    = self.sigmoid(x)
        result = sig * (1 - sig)
        return result
            
    def mse_loss(self , y_real , y_prediction):
        # y_real ve y_prediction aynı boyutta numpy arrayleri olmalıdır. 
        return ((y_real - y_prediction) ** 2).mean()
    
    ## İleri beslemeli nöronlar üzerinden tahmin
    ## değerinin elde edilmesi 
    
    def feedforward(self , row):        
        #h1 noron
        h1 = self.sigmoid((self.w1 * row[0]) + (self.w2 * row[1]) + self.b1 )
        
        #h2 noron
        h2 = self.sigmoid((self.w3 * row[0]) + (self.w4 * row[1]) + self.b2 )
        
        #tahmin degeri 01 noron sonucu
        o1 = self.sigmoid((self.w5 * h1 ) + (self.w6 * h2 ) + self.b3 )

        return o1 
    
    #belirtilen tekrar sayisi kadar model egitimi
    def train(self , trainingData , trainingLabels):
        learning_rate = 0.001
        epochs = 1000
        
        for epoch in range(epochs):
            for x, y in zip(trainingData , trainingLabels):
                #H1 noron
                sumH1 = (self.w1 * x[0]) + (self.w2 * x[1]) + self.b1 
                H1    = self.sigmoid(sumH1)
                
                #H2 noron
                sumH2 = (self.w3 * x[0]) + (self.w4 * x[1]) + self.b2
                H2    = self.sigmoid(sumH2)
                
                #01 noron
                sumO1 = (self.w5 * H1) + (self.w6 * H2) + self.b3
                O1    = self.sigmoid(sumO1)
                
                #tahmin
                prediction = O1
                
                # dL/dYpred :  y = dogru deger | prediciton: tahmin degeri
                dLoss_dPrediction = -2*(y - prediction)
                
                #H1 icin agirlik ve bias turevleri 
                dH1_dW1 = x[0] * self.sigmoid_turev(sumH1)
                dH1_dW2 = x[1] * self.sigmoid_turev(sumH1)
                dH1_dB1 = self.sigmoid_turev(sumH1)
                
                #H2 icin agirlik ve bias turevleri
                dH2_dW3 = x[0] * self.sigmoid_turev(sumH2)
                dH2_dW4 = x[1] * self.sigmoid_turev(sumH2)
                dH2_dB2 = self.sigmoid_turev(sumH2)
                
                #Noron O1 (output) icin agirlik ve bias turevleri
                dPrediction_dW5 = H1 * self.sigmoid_turev(sumO1) 
                dPrediction_dW6 = H2 * self.sigmoid_turev(sumO1) 
                dPrediction_dB3 = self.sigmoid_turev(sumO1) 
                
                #Tahmin degerinin H1 ve H2ye gore turevleri
                dPrediction_dH1 = self.w5 * self.sigmoid_turev(sumO1)
                dPrediction_dH2 = self.w6 * self.sigmoid_turev(sumO1)
                
                #H1 noronu
                self.w1 = self.w1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW1)
                self.w2 = self.w2 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW2)
                self.b1 = self.b1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dB1)
                
                #H2 noronu
                self.w3 = self.w3 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW3)
                self.w4 = self.w4 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW4)
                self.b2 = self.b2 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dB2)
                
                #output noronu
                self.w5 = self.w5 - (learning_rate * dLoss_dPrediction * dPrediction_dW5) 
                self.w6 = self.w6 - (learning_rate * dLoss_dPrediction * dPrediction_dW6) 
                self.b3 = self.b3 - (learning_rate * dLoss_dPrediction * dPrediction_dB3) 
                
            predictions = np.apply_along_axis(self.feedforward ,1, trainingData)
            loss = self.mse_loss(trainingLabels , predictions)
            print("Devir %d loss: %.7f" % (epoch, loss))
            
#Training dosyasinin acilimi
trainingFile = "Trainingset.csv"
try:
    trainingCsvFile = open(trainingFile, 'rt', encoding='utf-8-sig')
except:
    print("File not found")

csvReader = csv.reader(trainingCsvFile, delimiter=",")
l =  list()
cinsiyet = list()
for row in csvReader:
    l.append((int(float(row[0]))-160, int(float(row[1]))-71))
    cinsiyet.append(int(row[2]))

trainingData = np.asarray(l, dtype=np.float128)
trainingLabels = np.asarray(cinsiyet)

#Test verisetinin acilimi
testFile = 'Testset.csv'
try:
    testCsvFile = open(testFile, 'rt', encoding='utf-8-sig')
except:
    print('File not found')

testCsvReader = csv.reader(testCsvFile, delimiter=',')
t = list()
for row in testCsvReader:
    t.append((int(float(row[0]))-160, int(float(row[1]))-71, int(row[2])))

testData = np.asarray(t, dtype=np.float128)

#Sinir agi acilimi ve modelin egitimi
network = Network()
network.train(trainingData, trainingLabels)

#Test verisetindeki degerlerin tek tek sonuclari
accData = []
for x, y, z in testData:
    prediction = network.feedforward([x, y])
    if(prediction > 0.5): 
        print('Cinsiyet : Kadın  |' , 'Value : ' , prediction)
    else: 
        print('Cinsiyet : Erkek  |' , 'Value : ' , prediction)
    accData.append(round(prediction))

#Modelin isabet orani
accuracy = 0
for i in range(len(accData)):
    if accData[i] == testData[i][2]:
        accuracy += 1
print('Accuracy of model is %{:.5f}'.format((accuracy/len(accData))*100))