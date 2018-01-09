//
// Created by house on 5/3/17.
//

#include <cstdio>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <cstring>
#include <vector>

using namespace std;

#define MAX_DOUBLE 1e100
#define MIN_ACCURACY 1e-100
#define MAX_CHAR_PER_LINE 200000

#define PRINT_LOG true

vector<int> rightlog;

// train/test data
vector<vector<double> > data_x; // data_n * ts_len
vector<vector<double> > data_y; // data_n * num_c

int ts_len; // time series length
int data_n; // number of data
int num_c; // number of class

double alpha = -25.0;
double learn_rate = 0.01;
double regularization = 0.01;
int max_iter = 10000;

vector<double> w0;  // num_c
vector<vector<double> > w; // num_c * shapelet_n
vector<vector<vector<double> > > shapelet; // num_c * shapelet_n * shapelet_length
vector<vector<int> > shapelet_length; // num_c * shapelet_n
vector<int> shapelet_n; // number of shapelet for each class

vector<vector<vector<double> > > dis; // num_c * number of shapelet * match_offset
vector<vector<vector<double> > > e_dis; // num_c * number of shapelet * match_offset
vector<vector<double> > sum_e_dis; // num_c * number of shapelet
vector<vector<double> > m; // num_c * number of shapelet

int right_cnt = 0;

//*************************************************************

vector<vector<double> > test_x; // test_n * ts_len
vector<vector<double> > test_y; // test_n * num_c

int test_n; // number of data

void readtestfile(char *data_file) {
    FILE *f;
    f = fopen(data_file, "r");

    char buff[MAX_CHAR_PER_LINE];
    char *tmp;

    for (int i = 0; i < test_n; ++i) {
        test_x.push_back(vector<double>());
        test_y.push_back(vector<double>());
    }

    int *label = new int[test_n];
    int min_c = MAX_CHAR_PER_LINE;

    for (int i = 0; i < test_n; i++) {
        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");

        label[i] = atoi(tmp);
        min_c = min(min_c, label[i]);

        double sumx = 0;
        double sumx2 = 0;

        tmp = strtok(NULL, ", \r\n");
        for (int j = 0; j < ts_len; ++j) {
            test_x[i].push_back(atof(tmp));
            tmp = strtok(NULL, ", \r\n");
            sumx += test_x[i][j];
            sumx2 += test_x[i][j] * test_x[i][j];
        }

        double avgx = sumx / ts_len;
        double stdx = sqrt(sumx2 / ts_len - avgx * avgx);
        for (int j = 0; j < ts_len; ++j) {
            test_x[i][j] = (test_x[i][j] - avgx) / stdx;
        }
    }

    for (int i = 0; i < test_n; ++i) {
        label[i] -= min_c;
    }

    /**
     * label(xi) = a => y = {0 .... 1 .... 0}
     *                              ^
     *                              a position
     */
    for (int i = 0; i < test_n; ++i) {
        for (int j = 0; j < num_c; ++j) {
            if (j == label[i]) {
                test_y[i].push_back(1.0);
            } else {
                test_y[i].push_back(0.0);
            }
        }
    }

    delete[] label;

    fclose(f);
}

//*************************************************************

/**
 * read file data( n lines, len + 1 numbers each line )
 * first number is label, next len numbers are x
 * y[label] = 1, y[else] = 0
 */
void readfile(char *data_file) {
    FILE *f;
    f = fopen(data_file, "r");

    char buff[MAX_CHAR_PER_LINE];
    char *tmp;

    for (int i = 0; i < data_n; ++i) {
        data_x.push_back(vector<double>());
        data_y.push_back(vector<double>());
    }

    int *label = new int[data_n];
    int min_c = MAX_CHAR_PER_LINE;

    for (int i = 0; i < data_n; i++) {
        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");

        label[i] = atoi(tmp);
        min_c = min(min_c, label[i]);

        double sumx = 0;
        double sumx2 = 0;

        tmp = strtok(NULL, ", \r\n");
        for (int j = 0; j < ts_len; ++j) {
            data_x[i].push_back(atof(tmp));
            tmp = strtok(NULL, ", \r\n");
            sumx += data_x[i][j];
            sumx2 += data_x[i][j] * data_x[i][j];
        }

        double avgx = sumx / ts_len;
        double stdx = sqrt(sumx2 / ts_len - avgx * avgx);
        for (int j = 0; j < ts_len; ++j) {
            data_x[i][j] = (data_x[i][j] - avgx) / stdx;
        }
    }

    num_c = 0;
    for (int i = 0; i < data_n; ++i) {
        label[i] -= min_c;
        num_c = max(num_c, label[i]);
    }
    ++num_c;

    /**
     * label(xi) = a => y = {0 .... 1 .... 0}
     *                              ^
     *                              a position
     */
    for (int i = 0; i < data_n; ++i) {
        for (int j = 0; j < num_c; ++j) {
            if (j == label[i]) {
                data_y[i].push_back(1.0);
            } else {
                data_y[i].push_back(0.0);
            }
        }
    }

    delete[] label;

    fclose(f);
}

/**
 * init w0, w and shapelet
 * op == 0 init from init.txt
 *         else init from learned.txt
 */
void init(int op) {
    FILE *f;

    char buff[MAX_CHAR_PER_LINE];
    char *tmp;

    if (op == 0) {
        f = fopen("init.txt", "r");

        for (int i = 0; i < num_c; ++i) {
            w0.push_back(1.0);
        }
    } else {
        f = fopen("learned.txt", "r");

        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");
        for (int i = 0; i < num_c; ++i) {
            w0.push_back(atof(tmp));
            tmp = strtok(NULL, ", \r\n");
        }
    }

    for (int i = 0; i < num_c; ++i) {

        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");
        shapelet_n.push_back(atoi(tmp));
        tmp = strtok(NULL, ", \r\n");

        shapelet_length.push_back(vector<int>());
        shapelet.push_back(vector<vector<double> >());
        w.push_back(vector<double>());
        for (int j = 0; j < shapelet_n[i]; ++j) {
            fgets(buff, MAX_CHAR_PER_LINE, f);
            tmp = strtok(buff, ", \r\n");

            w[i].push_back(atof(tmp));
            tmp = strtok(NULL, ", \r\n");

            shapelet_length[i].push_back(atoi(tmp));
            tmp = strtok(NULL, ", \r\n");

            shapelet[i].push_back(vector<double>());
            for (int k = 0; k < shapelet_length[i][j]; ++k) {
                shapelet[i][j].push_back(atof(tmp));
                tmp = strtok(NULL, ", \r\n");
            }
        }
    }

    for (int i = 0; i < num_c; ++i) {

        dis.push_back(vector<vector<double> >());
        e_dis.push_back(vector<vector<double> >());
        sum_e_dis.push_back(vector<double>());
        m.push_back(vector<double>());

        for (int j = 0; j < shapelet_n[i]; ++j) {

            dis[i].push_back(vector<double>());
            e_dis[i].push_back(vector<double>());
            sum_e_dis[i].push_back(0.0);
            m[i].push_back(0.0);

            for (int k = 0; k <= ts_len - shapelet_length[i][j]; ++k) {
                dis[i][j].push_back(0.0);
                e_dis[i][j].push_back(0.0);
            }
        }
    }

    fclose(f);
}

void correct(double &x) {
    if (x < -MAX_DOUBLE) {
        x = -MAX_DOUBLE;
    }
    if (x > MAX_DOUBLE) {
        x = MAX_DOUBLE;
    }
    if (x > 0 && x < MIN_ACCURACY) {
        x = MIN_ACCURACY;
    }
    if (x < 0 && x > -MIN_ACCURACY) {
        x = -MIN_ACCURACY;
    }
}

void train(vector<double> &x, vector<double> &y, double regular) {

    /**
     * calculate distance between shapelet and time series
     * then get the min(distance) by softmin function
     */

    for (int i = 0; i < num_c; ++i) { // number of classes
        for (int j = 0; j < shapelet_n[i]; ++j) { // number_of_shapelet
            sum_e_dis[i][j] = 0;
            m[i][j] = 0;
            for (int k = 0; k <= ts_len - shapelet_length[i][j]; ++k) { // match_offset
                dis[i][j][k] = 0;
                for (int l = 0; l < shapelet_length[i][j]; ++l) { // shapelet_length
                    dis[i][j][k] += (x[k + l] - shapelet[i][j][l]) * (x[k + l] - shapelet[i][j][l]);
                }
                dis[i][j][k] /= shapelet_length[i][j];
                correct(dis[i][j][k]);
//                if (isnan(dis[i][j][k])) {
//                    printf("fuck dis\n");
//                    exit(1);
//                }
                e_dis[i][j][k] = exp(dis[i][j][k] * alpha);
                correct(e_dis[i][j][k]);
//                if (isnan(e_dis[i][j][k])) {
//                    printf("fuck e_dis\n");
//                    exit(1);
//                }
                sum_e_dis[i][j] += e_dis[i][j][k];
                correct(sum_e_dis[i][j]);
//                if (isnan(sum_e_dis[i][j])) {
//                    printf("fuck sum_e_dis\n");
//                    exit(1);
//                }
                m[i][j] += dis[i][j][k] * e_dis[i][j][k];
                correct(m[i][j]);
//                if (isnan(m[i][j])) {
//                    printf("fuck m\n");
//                    exit(1);
//                }
            }
            double fzc_rem = m[i][j];
            m[i][j] /= sum_e_dis[i][j];
            correct(m[i][j]);
//            if (isnan(m[i][j])) {
//                printf("fuck m %f sum %f\n", fzc_rem, sum_e_dis[i][j]);
//                for (int k = 0; k <= ts_len - shapelet_length[i][j]; ++k) {
//                    printf("dis %f\n", dis[i][j][k]);
//                }
//                exit(1);
//            }
        }
    }

    int bst_y = 0, true_y = 0;
    double bst = -1e8;
    double *v = new double[num_c];
    for (int i = 0; i < num_c; ++i) {
        double predict_y = w0[i]; // predict y TODO w0 ?
        for (int j = 0; j < shapelet_n[i]; ++j) {
            predict_y += w[i][j] * m[i][j];
        }
        double sigmoid = 1.0 / (1.0 + exp(-predict_y)); // sigmoid(predict y)
        v[i] = y[i] - sigmoid; // loss
//        if (isnan(v[i])) {
//            printf("y %f exp %f sigmod %f truey %f\n", predict_y, exp(-predict_y), sigmoid, y[i]);
//            exit(1);
//        }
        correct(v[i]);

//        if (PRINT_LOG) {
//            if (predict_y > bst) {
//                bst = predict_y;
//                bst_y = i;
//            }
//            if (y[i] > 0.5) {
//                true_y = i;
//            }
//        }

    }

//    if (PRINT_LOG) {
//        if (bst_y == true_y) {
//            ++right_cnt;
//        }
//    }

    for (int i = 0; i < num_c; ++i) {
        for (int j = 0; j < shapelet_n[i]; ++j) {

            double itmp = learn_rate * v[i] * w[i][j] / shapelet_length[i][j] / sum_e_dis[i][j];
//            if (isnan(itmp)) {
//                printf("v %f w %f len %d sum %f\n", v[i], w[i][j], shapelet_length[i][j], sum_e_dis[i][j]);
//                exit(1);
//            }

            double fzc_rem = w[i][j];
            w[i][j] += learn_rate * (v[i] * m[i][j] - 2.0 * regular * w[i][j] / data_n);
//            if (isnan(w[i][j])) {
//                printf("v %f m %f w %f x %f y %f\n", v[i], m[i][j], fzc_rem, v[i] * m[i][j],
//                       2.0 * regular * w[i][j] / data_n);
//                exit(1);
//            }

            if (itmp > -MIN_ACCURACY && itmp < MIN_ACCURACY) {
                continue;
            }
            correct(itmp);

            for (int l = 0; l <= ts_len - shapelet_length[i][j]; ++l) {

                double tmp = 2.0 * e_dis[i][j][l] * (1.0 + alpha * (dis[i][j][l] - m[i][j])) * itmp;

//                if (isnan(tmp)) {
//                    printf("e %f dis %f m %f itmp %f\n", e_dis[i][j][l], dis[i][j][l], m[i][j], itmp);
//                    exit(1);
//                }

                if (tmp > -MIN_ACCURACY && tmp < MIN_ACCURACY) {
                    continue;
                }
                correct(tmp);

                for (int p = 0; p < shapelet_length[i][j]; ++p) {
                    double fzc_rem = shapelet[i][j][p];
                    shapelet[i][j][p] += tmp * (shapelet[i][j][p] - x[l + p]);
//                    if (isnan(shapelet[i][j][p])) {
//                        printf("tmp %f shapelet %f x %f\n", tmp, fzc_rem, x[l + p]);
//                        exit(1);
//                    }
                }
            }
        }
        double fzc_rem = w0[i];
        w0[i] += learn_rate * v[i];
//        if (isnan(w0[i])) {
//            printf("w0 %f v %f\n", fzc_rem, v[i]);
//            exit(1);
//        }
    }

    delete[] v;
}

/**
 * save w0 , w and shapelet in learned.txt
 */
void save() {
    FILE *f;
    f = fopen("learned.txt", "w");

    for (int i = 0; i < num_c; ++i) {
        fprintf(f, "%f", w0[i]);
        if (i != num_c - 1) {
            fprintf(f, ",");
        } else {
            fprintf(f, "\n");
        }
    }

    for (int i = 0; i < num_c; ++i) {
        fprintf(f, "%d\n", shapelet_n[i]);
        for (int j = 0; j < shapelet_n[i]; ++j) {
            fprintf(f, "%f,%d", w[i][j], shapelet_length[i][j]);
            for (int k = 0; k < shapelet_length[i][j]; ++k) {
                fprintf(f, ",%f", shapelet[i][j][k]);
            }
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

void predict(vector<double> &x, vector<double> &y) {

    /**
     * calculate distance between shapelet and time series
     * then get the min(distance) by softmin function
     */

    for (int i = 0; i < num_c; ++i) { // number of classes
        for (int j = 0; j < shapelet_n[i]; ++j) { // number_of_shapelet
            sum_e_dis[i][j] = 0;
            m[i][j] = 0;
            for (int k = 0; k <= ts_len - shapelet_length[i][j]; ++k) { // match_offset
                dis[i][j][k] = 0;
                for (int l = 0; l < shapelet_length[i][j]; ++l) { // shapelet_length
                    dis[i][j][k] += (x[k + l] - shapelet[i][j][l]) * (x[k + l] - shapelet[i][j][l]);
                }
                dis[i][j][k] /= shapelet_length[i][j];
                correct(dis[i][j][k]);
                e_dis[i][j][k] = exp(dis[i][j][k] * alpha);
                correct(e_dis[i][j][k]);
                sum_e_dis[i][j] += e_dis[i][j][k];
                correct(sum_e_dis[i][j]);
                m[i][j] += dis[i][j][k] * e_dis[i][j][k];
                correct(m[i][j]);
            }
            m[i][j] /= sum_e_dis[i][j];
            correct(m[i][j]);
        }
    }

    // predict
    for (int i = 0; i < num_c; ++i) {
        double predict_y = w0[i]; // predict y TODO w0 ?
        for (int j = 0; j < shapelet_n[i]; ++j) {
            predict_y += w[i][j] * m[i][j];
        }
        y[i] = 1.0 / (1.0 + exp(-predict_y));
    }
}

int main(int argc, char *argv[]) {

    /**
     * arguments:
     * 1.train ts_len data_file data_n alpha regular max_iter learn_rate [test_data_file test_n]
     * 2.test ts_len data_file data_n alpha
     */
    char *operation = argv[1];
    ts_len = atoi(argv[2]);

    char *data_file = argv[3];
    data_n = atoi(argv[4]);

    alpha = atof(argv[5]);

    readfile(data_file);

    double start_time = clock();

    if (strcmp(operation, "train") == 0) {

        regularization = atof(argv[6]);
        max_iter = atoi(argv[7]);
        learn_rate = atof(argv[8]);

        if (argc == 11) {
            test_n = atoi(argv[10]);
            readtestfile(argv[9]);
        }

        init(0);

        // training
        bool first = true;
        for (int epoch = 0; epoch < max_iter; ++epoch) {
            if (epoch % 100 == 0)
                printf("epoch %d\n", epoch);
            for (int i = 0; i < data_n; ++i) {
                if (first) {
                    printf("data format:\n");
                    printf("data:\n");
                    for (int j = 0; j < ts_len; ++j) {
                        printf("%9.6f", data_x[i][j]);
                    }
                    printf("\n");
                    printf("label:\n");
                    for (int j = 0; j < num_c; ++j) {
                        printf("%9.0f", data_y[i][j]);
                    }
                    printf("\n");
                    first = false;
                }
                train(data_x[i], data_y[i], regularization);
            }

            if (PRINT_LOG) {
                rightlog.push_back(right_cnt);
//                printf("epoch %d accuracy %f\n", epoch, 1.0 * right_cnt / data_n);
                right_cnt = 0;

                if (argc == 11 && epoch % 100 == 0) {
                    int total_case = test_n;
                    int total_wrong = 0;

                    //testing
                    for (int i = 0; i < test_n; ++i) {

                        int true_y, pred_y = 0;
                        for (int j = 0; j < num_c; j++) {
                            if (test_y[i][j] > 0) {
                                true_y = j;
                            }
                            test_y[i][j] = 0; // clear y
                        }

                        predict(test_x[i], test_y[i]);

                        // find the max(y) and its index is the label of the case
                        for (int j = 1; j < num_c; j++) {
                            if (test_y[i][j] > test_y[i][pred_y]) {
                                pred_y = j;
                            }
                        }

                        if (true_y != pred_y) {
                            ++total_wrong;
                        }

                        // reset y
                        for (int j = 0; j < num_c; j++) {
                            test_y[i][j] = 0.0;
                            if (j == true_y) {
                                test_y[i][j] = 1.0;
                            }
                        }

                    }

                    printf("Accuracy = %8.3f Correct = %5d , Wrong = %5d\n",
                           100.0 * (total_case - total_wrong) / total_case, total_case - total_wrong, total_wrong);
                }

            }

        }

        if (PRINT_LOG) {
            FILE *flog;
            flog = fopen("rightlog", "w");
            for (int i = 0; i < max_iter; ++i) {
                fprintf(flog, "epoch %d accuracy %f\n", i + 1, 1.0 * rightlog[i] / data_n);
            }
            fclose(flog);
        }

        // output w, w0 and shapelet
        save();

    } else if (strcmp(operation, "test") == 0) {

        // input w, w0 and shapelet
        init(1);

        int total_case = data_n;
        int total_wrong = 0;
        FILE *f = fopen("result.txt", "a");

        //testing
        for (int i = 0; i < data_n; ++i) {

            int true_y, pred_y = 0;
            for (int j = 0; j < num_c; j++) {
                if (data_y[i][j] > 0) {
                    true_y = j;
                }
                data_y[i][j] = 0; // clear y
            }

            predict(data_x[i], data_y[i]);
            // find the max(y) and its index is the label of the case
            for (int j = 1; j < num_c; j++) {
                if (data_y[i][j] > data_y[i][pred_y]) {
                    pred_y = j;
                }
            }

            // a wrong case found
            if (true_y != pred_y) {
                ++total_wrong;
                fprintf(f, "%d case true label %d predict label %d\n", i + 1, true_y, pred_y);
                for (int j = 0; j < num_c; ++j) {
                    fprintf(f, "%9.6f", data_y[i][j]);
                    if (j != num_c - 1) {
                        fprintf(f, ",");
                    } else {
                        fprintf(f, "\n");
                    }
                }
            }

        }

        fprintf(f, "Accuracy = %8.3f Correct = %5d , Wrong = %5d\n", 100.0 * (total_case - total_wrong) / total_case,
                total_case - total_wrong, total_wrong);
        fclose(f);

    } else {
        printf("wrong operation !\n");
    }

    double end_time = clock();
    printf("Total Running Time = %.3f sec\n", (end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}