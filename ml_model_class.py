class Linear_Regression_Ml:

    def init(self):
        self.slope = None
        self.intercept = None
        self.predicted_value = None
        self.avg_squared_error = None
        self.r2_value = None

    def fit(self, x_train, y_train):

        train_length = len(x_train)
        sum_x_train=0
        sum_y_train=0
        sum_x_diff_mean_x_mult_y_diff_mean_y = 0
        sum_x_diff_mean_x_squared = 0


        for i in range(train_length):
            sum_x_train += x_train[i]
            sum_y_train += y_train[i]

        mean_x = sum_x_train/(train_length)
        mean_y = sum_y_train/(train_length)


        for i in range(train_length):
            sum_x_diff_mean_x_mult_y_diff_mean_y += (x_train[i]-mean_x)*(y_train[i]-mean_y)
            sum_x_diff_mean_x_squared += ((x_train[i]-mean_x) **2)

        self.slope = sum_x_diff_mean_x_mult_y_diff_mean_y/sum_x_diff_mean_x_squared
        self.intercept = mean_y - (self.slope*mean_x)

        return self


    def predict(self, x_input):
        x_input = float(input("Enter the hours studied by student: "))
        self.predicted_value = self.intercept + (self.slope * x_input)
        return self


    def ml_mse_r2_score(self, x_test, y_test):
        sum_y_diff_mean_y_squared = 0

        length = len(x_test)

        y_predict = []
        squared_error_sum = 0
        sum_y_test=0

            
        for i in range(length):
            y_predict.append(self.intercept + (self.slope * x_test[i]))
            squared_error_sum += ((y_test[i]-y_predict[i]) **2)
            sum_y_test += y_test[i]
            
        mean_y_test = sum_y_test/(len(y_test))

        for data in y_test:
            sum_y_diff_mean_y_squared += ((data - mean_y_test) **2)

        
        self.avg_squared_error = squared_error_sum/(length) 
        sse_div_sum_y_diff_mean_y_squared= squared_error_sum/sum_y_diff_mean_y_squared
        self.r2_value = 1-sse_div_sum_y_diff_mean_y_squared

        return self

        