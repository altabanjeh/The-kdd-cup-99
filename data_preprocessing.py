import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

start_time = time.time()  # Get the current time in seconds since the epoch

# Your process or code block goes here
# Replace this with your actual code


def data_preprocessing():
    header = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome'
    ]

    ######################################################################
    # read csv file

    data = pd.read_csv('data.csv', chunksize=10000,
                       on_bad_lines='skip', names=header)
    data1 = pd.concat(data)

    ######################################################################
    # check for duplicates

    data2 = data1.drop_duplicates()

    ######################################################################
    # splot the data to x and y
    x = data2.iloc[:, :41]
    y = data2.iloc[:, -1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    tenc = ce.TargetEncoder()
    encoder1 = ce.TargetEncoder(cols=['protocol_type'])
    encoder2 = ce.TargetEncoder(cols=['service'])
    encoder3 = ce.TargetEncoder(cols=['flag'])

    X_train_encoded = encoder1.fit_transform(x, y)
    X_test_encoded = encoder1.transform(x)
    X_train_encoded = encoder2.fit_transform(x, y)
    X_test_encoded = encoder2.transform(x)
    X_train_encoded = encoder3.fit_transform(x, y)
    X_test_encoded = encoder3.transform(x)
    data20 = x
    data20['protocol_type1'] = tenc.fit_transform(x['protocol_type'], y)
    data20.drop(columns=['protocol_type'], inplace=True)
    data20['service1'] = tenc.fit_transform(x['service'], y)
    data20.drop(columns=['service'], inplace=True)
    data20['flag1'] = tenc.fit_transform(x['flag'], y)
    data20.drop(columns=['flag'], inplace=True)

    last_two_columns = x.iloc[:, -3:]  # for get the right order for spilt X and Y
    rest_of_dataframe = x.iloc[:, :-3]
    x_reordered = pd.concat([last_two_columns, rest_of_dataframe], axis=1)
    x_reordered = np.append(
        np.ones((1074991, 1)).astype(int), x_reordered, axis=1)

    def multiple_reg(X, y):
        columns = list(range(X.shape[1]))
        for i in range(X.shape[1]):
            X_opt = np.array(X[:, list(columns)], dtype=float)
            regessor_ols = sm.OLS(endog=y, exog=X_opt).fit()
            pvalues = list(regessor_ols.pvalues)

            if(max(pvalues) > 0.05):
                for j in range(len(pvalues)):
                    if(pvalues[j] == max(pvalues)):
                        print(j)
                        del(columns[j])
                        break
                    return(X_opt)
            else:
                break

    x_opt = multiple_reg(x_reordered, y)

    print(pd.unique(y))

    x_opt_train, x_opt_test, y_train, y_test = train_test_split(
        x_opt, y, test_size=0.2, random_state=1)

    sc = StandardScaler()
    x_opt_train = sc.fit_transform(x_opt_train)
    x_opt_test = sc.transform(x_opt_test)

    end_time = time.time()  # Get the current time again

    elapsed_time = end_time - start_time

    t1 = elapsed_time/60
    if t1 < 1:
        t1 = elapsed_time
        print("Elapsed time:", t1, "second ")
    else:
        print("Elapsed time:", t1, "minutes")

    return (x_opt_train, x_opt_test, y_train, y_test, t1)


# Define x_opt_train outside of the function
x_opt_train, x_opt_test, y_train, y_test, t1 = data_preprocessing()
