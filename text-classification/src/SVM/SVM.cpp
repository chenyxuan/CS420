/*
	Author: Chen Yuxuan
	Problem: binary classification
	Solution: SVM without kernel method
*/

#include <bits/stdc++.h>

class SVM {
	typedef std::pair<double *,double> Data;
	static const double eps = 1e-8;

private:
	int m;
	std::vector<double> a;
	double *c;
	
	std::vector<Data> data;
	int dims;
	
	int cmp(double x,double y) {
		if(fabs(x - y) < eps) return 0;
		return x < y ? -1 : 1;
	}
	
	double mul(double *P,double *Q) {
		double res = 0;
		
		for(int i = 0; i < dims; i++) {
			res += P[i] * Q[i];
		}		
		return res;
	}
	
	bool is_valid() {
		int postive_cnt = 0;
		int negative_cnt = 0;
		
		for(int i = 0; i < m; i++) {
			double y_i = data[i].second;
			
			if(cmp(y_i, 0) > 0) postive_cnt++;
			else if(cmp(y_i, 0) < 0) negative_cnt++;
			else {
				/* label == 0 --> invalid data*/
				return false;
			}
		}
		
		return postive_cnt > 0 && negative_cnt > 0;
	}
	
	void build() {
		a.resize(m);
		std::fill(a.begin(), a.end(), 0);
		
		c = new double[m * m];
		
		double *c_now = c;
		
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < m; j++) {
				double *x_i = data[i].first;
				double *x_j = data[j].first;
				double y_i = data[i].second;
				double y_j = data[j].second;
				
				*c_now = y_i * y_j * mul(x_i, x_j);
				c_now++;
			}
		}
	}
	
	void destory() {
		a.resize(0);
		delete[] c;
	}

	double elapsed_time(double s_time) {
		return clock() / CLOCKS_PER_SEC - s_time;
	}
	
	double c_get(int x,int y) {
		return *(c + x * m + y);
	}
	
	double f(double x,double A,double B) {
		return A * x * x + B * x;
	}
	
	void smo_slack(std::vector<Data> &data, double C) {
		int p, q;
		
		do {
			p = rand() % m;
			q = rand() % m;	
		}while(p == q);
		
		double sum = 0;
		double y_p = data[p].second;
		double y_q = data[q].second;
		
		for(int i = 0; i < m; i++) {
			double y_i = data[i].second;
			
			if(i == p || i == q) continue;
			sum += a[i] * y_i;
		}
		
		double A = c_get(p, p), B = c_get(q, q);
		double D = c_get(p, q) + c_get(q, p);
		double F = -y_p / y_q, G = -sum / y_q;
		
		double P = -(A + D * F + B * F * F) / 2;
		double Q = -(2 * B * F * G + D * G) / 2 + (1 + F);
		
		double l = 0, r = C;
		
		if(cmp(F, 0) > 0) {
			l = std::max(l, -G / F);
			r = std::min(r, (C - G) / F);
		}
		else {
			l = std::max(l, (C - G) / F);
			r = std::min(r, -G / F);
		}
		
		assert(cmp(l, r) <= 0);
		
		double best = (f(l, P, Q) < f(r, P, Q) ? r : l);
		double eval = -Q / (2 * P);
		
		if(cmp(l, eval) <= 0 && cmp(eval, r) <= 0) {
			if(f(best, P, Q) < f(eval, P, Q)) best = eval;
		}		
		
		a[p] = best;
		a[q] = F * a[p] + G;
	}
	
	Data get_answer() {
		double *array = new double[dims];
		
		for(int i = 0; i < dims; i++) array[i] = 0;
		for(int i = 0; i < m; i++) {
			double *x_i = data[i].first;
			double y_i = data[i].second;
			
			for(int j = 0; j < dims; j++) {
				array[j] += a[i] * y_i * x_i[j];
			}
		}
		
		double bmax, bmin;
		bool bmax_init = false;
		bool bmin_init = false;
		
		for(int i = 0; i < m; i++) {
			double *x_i = data[i].first;
			double y_i = data[i].second;
			
			double upd = mul(array, x_i);
			
			if(cmp(y_i, 0) < 0) {
				if(!bmax_init) {
					bmax = upd;
					bmax_init = true;
				}
				else {
					bmax = std::max(bmax, upd);
				}
			}
			else {
				if(!bmin_init) {
					bmin = upd;
					bmin_init = true;
				}
				else {
					bmin = std::min(bmin, upd);
				}
			}
		}
		
		assert(bmax_init && bmin_init);
		
		return std::make_pair(array, - (bmax + bmin) / 2);
	}
	
public:
 	Data solve(std::vector<Data> data, int dims, double C, double T = 10) {
		/*
			data[i].first  := feature of instance i
			data[i].second := label of instance i
			
			T := execute T seconds
		*/
		
		/* init */
		this->m = data.size();
		this->dims = dims;
		this->data.clear();
		this->data.assign(data.begin(), data.end());

		assert(is_valid());

		build();
		srand(time(NULL));
		
		double start_time = clock() / CLOCKS_PER_SEC;
		while(elapsed_time(start_time) < T) smo_slack(data, C);
		
		Data res = get_answer();

		destory();
		return res;
	}
	
};

int main() {
	int n, m;
	double C;
	std::vector<std::pair<double*,double> > input;

	scanf("%d%d%lf", &n, &m, &C);
	for(int i = 0; i < n; i++) {
		double *feature = new double[m];
		double label = 0;
		
		for(int j = 0; j < m; j++) {
			scanf("%lf", feature + j);
		}
		scanf("%lf", &label);
		
		input.push_back(std::make_pair(feature, label));
	}
	
	SVM svm;
	std::pair<double*, double> res = svm.solve(input, m, C, 1);
	for(int i = 0; i < m; i++) {
		printf("%lf%c", *(res.first + i), " \n"[i == m - 1]);
	}
	printf("%lf\n", res.second);
	
	return 0;
}
