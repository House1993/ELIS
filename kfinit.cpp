//
// Created by house on 4/14/17.
//

#include <cstdio>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <set>
#include <map>

using namespace std;

#define RTREE_DIMENSION 10
#define PAA_BUCKET 10
#define inf 200000

#define COVER_SAME 1
#define PAA_SAME 0.1
#define PAA_LIKE 0.2
#define PAA_DIFF 2

#define MAX_CHAR_PER_LINE 200000
#define MAX_CLASS 100

#define x2_0500 3.84
#define x2_0250 5.02
#define x2_0200 5.41
#define x2_0100 6.63
#define x2_0050 7.88
#define x2_0025 9.14
#define x2_0010 10.83
#define x2_0005 12.12

// train/test data
vector<vector<double> > data_x; // data_n * ts_len
double max_data = -inf, min_data = inf, bucket_w;
vector<int> label;
int datacnt[MAX_CLASS];
int shapelets_n[MAX_CLASS];

int ts_len; // time series length
int data_n; // number of data
int num_c, min_class_id = inf; // number of class , min class id (init inf)

struct point {
    int f[RTREE_DIMENSION];

    bool operator<(const point ff) const {
        for (int i = 0; i < RTREE_DIMENSION - 1; ++i) {
            if (f[i] != ff.f[i]) {
                return f[i] < ff.f[i];
            }
        }
        return f[RTREE_DIMENSION - 1] < ff.f[RTREE_DIMENSION - 1];
    }
};

map<point, set<int> > tree[MAX_CLASS];

struct candidatesax {
    double score;
    int sign;
    int window;
    vector<int> vec;
    set<int> cov;
    int a, b, c, d;
    double ta, tb, tc, td;

//    bool operator<(const candidatesax f) const { // big head
//        return score < f.score;
//    }

    bool operator<(const candidatesax f) const { // for set
        if (window != f.window) {
            return window < f.window;
        }
        for (int i = 0; i < RTREE_DIMENSION; ++i) {
            if (vec[i] < f.vec[i] - 1 || vec[i] > f.vec[i] + 1) {
                return vec[i] < f.vec[i];
            }
        }
        return false;
    }

    bool operator==(const candidatesax f) const {
        if (f.window * 2 <= window) {
            return false;
        }
        int mincnt = f.window * RTREE_DIMENSION;
        for (int offset = 0; offset <= window * RTREE_DIMENSION - f.window * RTREE_DIMENSION; ++offset) {
            int cnt = 0;
            for (int i = window - offset % window, di = offset / window, j = f.window, dj = 0;
                 dj < RTREE_DIMENSION;) {
                int num = min(i, j);
                if (vec[di] < f.vec[dj] - 1 || vec[di] > f.vec[dj] + 1) {
                    if (vec[di] < f.vec[dj] - PAA_DIFF || vec[di] > f.vec[dj] + PAA_DIFF) {
                        cnt = f.window * RTREE_DIMENSION;
                        break;
                    }
                    cnt += num;
                    if (cnt > PAA_LIKE * RTREE_DIMENSION * f.window) {
                        break;
                    }
                }
                i -= num;
                j -= num;
                if (i == 0) {
                    i = window;
                    ++di;
                }
                if (j == 0) {
                    j = f.window;
                    ++dj;
                }
            }
            if (cnt <= PAA_SAME * RTREE_DIMENSION * f.window) {
                //-----------------------------------------------------------------------
                printf("-------------------PAA SAME-------------------\n");
                printf("%f,%d", score * sign, window * RTREE_DIMENSION);
                for (int k = 0; k < RTREE_DIMENSION; ++k) {
                    printf(",%d", vec[k]);
                }
                printf("\ncover");
                for (set<int>::iterator x = cov.begin(); x != cov.end(); ++x) {
                    printf(" %d", *x);
                }
                printf("\n");

                printf("%f,%d", f.score * f.sign, f.window * RTREE_DIMENSION);
                for (int k = 0; k < RTREE_DIMENSION; ++k) {
                    printf(",%d", f.vec[k]);
                }
                printf("\ncover");
                for (set<int>::iterator x = f.cov.begin(); x != f.cov.end(); ++x) {
                    printf(" %d", *x);
                }
                printf("\n");
                printf("-----------------------------------------------\n");
                //-----------------------------------------------------------------------
                return true;
            }
            mincnt = min(mincnt, cnt);
        }
        if (mincnt > PAA_LIKE * RTREE_DIMENSION * f.window) {
            return false;
        }
        int cnt = 0;
        for (set<int>::iterator t1 = cov.begin(), t2 = f.cov.begin(); t1 != cov.end() && t2 != f.cov.end();) {
            if ((*t1) == (*t2)) {
                ++t1;
                ++t2;
                ++cnt;
            } else if ((*t1) < (*t2)) {
                ++t1;
            } else {
                ++t2;
            }
        }

        if (cnt >= cov.size() * COVER_SAME && cnt >= f.cov.size() * COVER_SAME) {

            //-----------------------------------------------------------------------
            printf("----------------PAA COVER LIKE----------------\n");
            printf("%f,%d", score * sign, window * RTREE_DIMENSION);
            for (int k = 0; k < RTREE_DIMENSION; ++k) {
                printf(",%d", vec[k]);
            }
            printf("\ncover");
            for (set<int>::iterator x = cov.begin(); x != cov.end(); ++x) {
                printf(" %d", *x);
            }
            printf("\n");

            printf("%f,%d", f.score * f.sign, f.window * RTREE_DIMENSION);
            for (int k = 0; k < RTREE_DIMENSION; ++k) {
                printf(",%d", f.vec[k]);
            }
            printf("\ncover");
            for (set<int>::iterator x = f.cov.begin(); x != f.cov.end(); ++x) {
                printf(" %d", *x);
            }
            printf("\n");
            printf("-----------------------------------------------\n");
            //-----------------------------------------------------------------------

            return true;
        }
        return false;
    }

//    bool operator==(const candidatesax f) const {
//        if (f.window * 2 <= window) {
//            return false;
//        }
//        for (int offset = 0; offset <= window * RTREE_DIMENSION - f.window * RTREE_DIMENSION; ++offset) {
//            int cnt = 0;
//            for (int i = window - offset % window, di = offset / window, j = f.window, dj = 0;
//                 dj < RTREE_DIMENSION;) {
//                int num = min(i, j);
//                if (vec[di] < f.vec[dj] - 1 || vec[di] > f.vec[dj] + 1) {
//                    if (vec[di] < f.vec[dj] - PAA_DIFF || vec[di] > f.vec[dj] + PAA_DIFF) {
//                        cnt = f.window * RTREE_DIMENSION;
//                        break;
//                    }
//                    cnt += num;
//                    if (cnt > PAA_SAME * RTREE_DIMENSION * f.window) {
//                        break;
//                    }
//                }
//                i -= num;
//                j -= num;
//                if (i == 0) {
//                    i = window;
//                    ++di;
//                }
//                if (j == 0) {
//                    j = f.window;
//                    ++dj;
//                }
//            }
//            if (cnt <= PAA_SAME * RTREE_DIMENSION * f.window) {
//                //-----------------------------------------------------------------------
//                printf("-------------------PAA SAME-------------------\n");
//                printf("%f,%d", score * sign, window * RTREE_DIMENSION);
//                for (int k = 0; k < RTREE_DIMENSION; ++k) {
//                    printf(",%d", vec[k]);
//                }
//                printf("\ncover");
//                for (set<int>::iterator x = cov.begin(); x != cov.end(); ++x) {
//                    printf(" %d", *x);
//                }
//                printf("\n");
//
//                printf("%f,%d", f.score * f.sign, f.window * RTREE_DIMENSION);
//                for (int k = 0; k < RTREE_DIMENSION; ++k) {
//                    printf(",%d", f.vec[k]);
//                }
//                printf("\ncover");
//                for (set<int>::iterator x = f.cov.begin(); x != f.cov.end(); ++x) {
//                    printf(" %d", *x);
//                }
//                printf("\n");
//                printf("-----------------------------------------------\n");
//                //-----------------------------------------------------------------------
//                return true;
//            }
//        }
//        return false;
//    }

};

set<candidatesax> que[MAX_CLASS];
set<candidatesax>::iterator it;

vector<candidatesax> buffer;

/**
 * read file data( n lines, len + 1 numbers each line )
 * first number is label, next len numbers are x
 */
void readfile(char *data_file) {
    FILE *f;
    f = fopen(data_file, "r");

    char buff[MAX_CHAR_PER_LINE];
    char *tmp;

    for (int i = 0; i < data_n; ++i) {
        data_x.push_back(vector<double>());
    }

    num_c = 0;

    for (int i = 0; i < data_n; i++) {
        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");

        label.push_back(atoi(tmp));

        min_class_id = min(min_class_id, label[i]);

        tmp = strtok(NULL, ", \r\n");
        for (int j = 0; j < ts_len; ++j) {

            double now_data = atof(tmp);
            max_data = max(max_data, now_data);
            min_data = min(min_data, now_data);

            data_x[i].push_back(now_data);
            tmp = strtok(NULL, ", \r\n");
        }
    }

    for (int i = 0; i < data_n; ++i) {
        label[i] -= min_class_id;
        num_c = max(num_c, label[i]);
    }

    ++num_c;

    for (int i = 0; i < data_n; ++i) {
        ++datacnt[label[i]];
    }

    bucket_w = (max_data - min_data) / PAA_BUCKET;
    max_data += bucket_w / 2;
    bucket_w = (max_data - min_data) / PAA_BUCKET;

    fclose(f);
}

void insertintortree(vector<int> &qu, int offset, int tsid) {
    point f;
    for (int i = 0; i < RTREE_DIMENSION; ++i) {
        f.f[i] = qu[offset + i];
    }
    tree[label[tsid]][f].insert(tsid);
    tree[num_c][f].insert(tsid);
}

double x2test(double a, double b, double c, double d) {

    double ta = (a + b) * (a + c) / data_n;
    double tb = (a + b) * (b + d) / data_n;
    double tc = (a + c) * (c + d) / data_n;
    double td = (b + d) * (c + d) / data_n;

    double tmp = a * d - b * c;

    if (ta > 5 && tb > 5 && tc > 5 && td > 5) {
        return tmp * fabs(tmp) * data_n / ((a + b) * (a + c) * (c + d) * (b + d));
    }

    if (tmp < 0) {
        tmp = -tmp - data_n / 2;
        return -(tmp * tmp * data_n / ((a + b) * (a + c) * (c + d) * (b + d)));
    }

    tmp = fabs(tmp) - data_n / 2;
    return tmp * tmp * data_n / ((a + b) * (a + c) * (c + d) * (b + d));
}

set<int> queryfromrtree(vector<int> &qu, int offset, int treeid) {
    set<int> cnt;
    point f;
    for (int i = 0; i < RTREE_DIMENSION; ++i) {
        f.f[i] = qu[offset + i];
    }
    if (tree[treeid].count(f)) {
        cnt.insert(tree[treeid][f].begin(), tree[treeid][f].end());
    }
    for (int i = 0; i < RTREE_DIMENSION; ++i) {
        f.f[i] -= 1;
        if (tree[treeid].count(f)) {
            cnt.insert(tree[treeid][f].begin(), tree[treeid][f].end());
        }
        f.f[i] += 2;
        if (tree[treeid].count(f)) {
            cnt.insert(tree[treeid][f].begin(), tree[treeid][f].end());
        }
        f.f[i] -= 1;
    }
    return cnt;
}

double x2table(double v) {
    if (v > x2_0005) {
        return 0.9995;
    } else if (v > x2_0010) {
        return 0.9990;
    } else if (v > x2_0025) {
        return 0.9975;
    } else if (v > x2_0050) {
        return 0.9950;
    } else if (v > x2_0100) {
        return 0.9900;
    } else if (v > x2_0200) {
        return 0.9800;
    } else if (v > x2_0250) {
        return 0.9750;
    } else if (v > x2_0500) {
        return 0.9500;
    } else {
        return 0;
    }
}

void findbestshape() {

    int sumvec = 0;

    int shapelet_minlen = max(RTREE_DIMENSION, (ts_len / 20 + RTREE_DIMENSION - 1) / RTREE_DIMENSION * RTREE_DIMENSION);

    for (int len = shapelet_minlen; len <= ts_len; len += shapelet_minlen) {

        printf("search length %d:\n", len);

        // clear rtree
        for (int i = 0; i <= num_c; ++i) {
            tree[i].clear();
        }

        printf("Hash map clear\n");

        // PAA window
        int window = len / RTREE_DIMENSION;

        for (int i = 0; i < data_n; ++i) {
            for (int start = 0; start < window; ++start) {
                vector<int> qu;

                // PAA
                for (int offset = start; offset + window <= ts_len; offset += window) {
                    double avg = 0;
                    for (int j = offset; j < offset + window; ++j) {
                        avg += data_x[i][j];
                    }
                    int bucket = (int) floor((avg / window - min_data) / bucket_w);
                    qu.push_back(bucket);
                }

                // insert into hash map
                for (int j = 0; j + RTREE_DIMENSION <= qu.size(); ++j) {
                    insertintortree(qu, j, i);
                }

            }
        }

        printf("insert finished\n");

        for (int i = 0; i < data_n; ++i) {
            for (int start = 0; start < window; ++start) {
                vector<int> qu;

                // PAA
                for (int offset = start; offset + window <= ts_len; offset += window) {
                    double avg = 0;
                    for (int j = offset; j < offset + window; ++j) {
                        avg += data_x[i][j];
                    }
                    int bucket = (int) floor((avg / window - min_data) / bucket_w);
                    qu.push_back(bucket);
                }

                // query from hash map
                for (int j = 0; j + RTREE_DIMENSION <= qu.size(); ++j) {
                    int seleceall = queryfromrtree(qu, j, num_c).size();
                    for (int k = 0; k < num_c; ++k) {
                        set<int> covts = queryfromrtree(qu, j, k);
                        int selectc = covts.size();
                        // TODO multi method
                        double score = x2test(selectc, seleceall - selectc, datacnt[k] - selectc,
                                              data_n - seleceall - datacnt[k] + selectc);
                        // TODO condition
                        if (fabs(score) > x2_0010) {
                            candidatesax now;
                            now.score = x2table(fabs(score));
                            now.sign = 1;
                            if (score < 0) {
                                now.sign = -1;
                            }
                            now.window = window;
                            for (int cp = 0; cp < RTREE_DIMENSION; ++cp) {
                                now.vec.push_back(qu[j + cp]);
                            }

                            now.cov = covts;

                            now.a = selectc;
                            now.b = seleceall - selectc;
                            now.c = datacnt[k] - selectc;
                            now.d = data_n - seleceall - datacnt[k] + selectc;

                            now.ta = (double) (now.a + now.b) * (now.a + now.c) / data_n;
                            now.tb = (double) (now.a + now.b) * (now.b + now.d) / data_n;
                            now.tc = (double) (now.a + now.c) * (now.c + now.d) / data_n;
                            now.td = (double) (now.b + now.d) * (now.c + now.d) / data_n;

                            que[k].insert(now);
//                            que[k].push(now);
//                            ++shapelets_n[k];
                        }
                    }
                }
            }
        }

        printf("query finished\n");

        sumvec += tree[num_c].size();

    }
    printf("total distinct subsequence %d\n", sumvec);
}

void printshape(char *out_file) {
    FILE *f;
    f = fopen(out_file, "w");

    for (int i = 0; i < num_c; ++i) {
        shapelets_n[i] = 0;
        buffer.clear();

        it = que[i].begin();
        while (it != que[i].end()) {

            vector<candidatesax> curlen;
            do {
                candidatesax now = (*it);
                curlen.push_back(now);
                ++it;
            } while (it != que[i].end() && it->window == curlen[0].window);

            for (int j = 0; j < curlen.size(); ++j) {
                bool insertcandidate = true;
//                for (int k = 0; k < buffer.size(); ++k) {
//                    if (curlen[j] == buffer[k]) {
//                        insertcandidate = false;
//                        break;
//                    }
//                }
                if (insertcandidate) {
                    ++shapelets_n[i];
                    buffer.push_back(curlen[j]);
                }
            }

//        for (int j = 0; j < shapelets_n[i] && !que[i].empty(); ++j) {
//            candidatesax now = que[i].top();
//            que[i].pop();

        }

        fprintf(f, "%d\n", shapelets_n[i]);
        for (int j = 0; j < shapelets_n[i]; ++j) {
            candidatesax &now = buffer[j];
            fprintf(f, "%f,%d", now.score * now.sign, now.window * RTREE_DIMENSION);

//            for (int k = 0; k < now.window * RTREE_DIMENSION; ++k) {
//                fprintf(f, ",%f", bucket_w * now.vec[k / now.window] + min_data);
//            }
//            fprintf(f, "\n");
            for (int k = 0; k < RTREE_DIMENSION; ++k) {
                fprintf(f, ",%d", now.vec[k]);
            }
            fprintf(f, "\ncover");
            for (set<int>::iterator cov = now.cov.begin(); cov != now.cov.end(); ++cov) {
                fprintf(f, " %d", *cov);
            }
            fprintf(f, "\n");
            fprintf(f, "%d(%f)\t\t%d(%f)\n", now.a, now.ta, now.b, now.tb);
            fprintf(f, "%d(%f)\t\t%d(%f)\n", now.c, now.tc, now.d, now.td);
        }
    }

    fclose(f);
}

int main(int argc, char *argv[]) {

    ts_len = atoi(argv[1]);
    char *data_file = argv[2];
    data_n = atoi(argv[3]);

    readfile(data_file);

    printf("reading data finished\n");

    double start_time = clock();
    findbestshape();
    double end_time = clock();

    printf("algorithm finished, start outputing\n");

    printshape("init.txt");

    printf("Total Running Time = %.3f sec\n", (end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}