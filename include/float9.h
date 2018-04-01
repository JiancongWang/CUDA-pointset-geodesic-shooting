#ifndef FLOAT9_H_
#define FLOAT9_H_

struct float9{
	float e00;
	float e01;
	float e02;

	float e10;
	float e11;
	float e12;

	float e20;
	float e21;
	float e22;


	void fill(float filler){
		this->e00 = filler;
		this->e01 = filler;
		this->e02 = filler;

		this->e10 = filler;
		this->e11 = filler;
		this->e12 = filler;

		this->e20 = filler;
		this->e21 = filler;
		this->e22 = filler;
	}


	inline void operator=(float9 const & another) {
		this->e00 = another.e00;
		this->e01 = another.e01;
		this->e02 = another.e02;

		this->e10 = another.e10;
		this->e11 = another.e11;
		this->e12 = another.e12;

		this->e20 = another.e20;
		this->e21 = another.e21;
		this->e22 = another.e22;
	}

	inline bool operator==(float9 const & another) {
		return  (this->e00 == another.e00)&&
				(this->e01 == another.e01)&&
				(this->e02 == another.e02)&&
				(this->e10 == another.e10)&&
				(this->e11 == another.e11)&&
				(this->e12 == another.e12)&&
				(this->e20 == another.e20)&&
				(this->e21 == another.e21)&&
				(this->e22 == another.e22);
	}

	inline void operator+=(float9 const & another) {
		this->e00+= another.e00;
		this->e01+= another.e01;
		this->e02+= another.e02;

		this->e10+= another.e10;
		this->e11+= another.e11;
		this->e12+= another.e12;

		this->e20+= another.e20;
		this->e21+= another.e21;
		this->e22+= another.e22;
	}

	inline void operator-=(float9 const & another) {
		this->e00-= another.e00;
		this->e01-= another.e01;
		this->e02-= another.e02;

		this->e10-= another.e10;
		this->e11-= another.e11;
		this->e12-= another.e12;

		this->e20-= another.e20;
		this->e21-= another.e21;
		this->e22-= another.e22;
	}

	inline void operator*=(float9 const & another) {
		this->e00*= another.e00;
		this->e01*= another.e01;
		this->e02*= another.e02;

		this->e10*= another.e10;
		this->e11*= another.e11;
		this->e12*= another.e12;

		this->e20*= another.e20;
		this->e21*= another.e21;
		this->e22*= another.e22;
	}

	inline void operator/=(float9 const & another) {
		this->e00/= another.e00;
		this->e01/= another.e01;
		this->e02/= another.e02;

		this->e10/= another.e10;
		this->e11/= another.e11;
		this->e12/= another.e12;

		this->e20/= another.e20;
		this->e21/= another.e21;
		this->e22/= another.e22;
	}

	inline float9 operator+(float9 const & b) {
		float9 s;
		s.e00 = this->e00 + b.e00;
		s.e01 = this->e01 + b.e01;
		s.e02 = this->e02 + b.e02;

		s.e10 = this->e10 + b.e10;
		s.e11 = this->e11 + b.e11;
		s.e12 = this->e12 + b.e12;

		s.e20 = this->e20 + b.e20;
		s.e21 = this->e21 + b.e21;
		s.e22 = this->e22 + b.e22;
		return s;
	}

	inline float9 operator-(float9 const & b) {
		float9 s;
		s.e00 = this->e00 - b.e00;
		s.e01 = this->e01 - b.e01;
		s.e02 = this->e02 - b.e02;

		s.e10 = this->e10 - b.e10;
		s.e11 = this->e11 - b.e11;
		s.e12 = this->e12 - b.e12;

		s.e20 = this->e20 - b.e20;
		s.e21 = this->e21 - b.e21;
		s.e22 = this->e22 - b.e22;
		return s;
	}

	inline float9 operator*(float9 const & b) {
		float9 s;
		s.e00 = this->e00 * b.e00;
		s.e01 = this->e01 * b.e01;
		s.e02 = this->e02 * b.e02;

		s.e10 = this->e10 * b.e10;
		s.e11 = this->e11 * b.e11;
		s.e12 = this->e12 * b.e12;

		s.e20 = this->e20 * b.e20;
		s.e21 = this->e21 * b.e21;
		s.e22 = this->e22 * b.e22;
		return s;
	}

	inline float9 operator/(float9 const & b) {
		float9 s;
		s.e00 = this->e00 / b.e00;
		s.e01 = this->e01 / b.e01;
		s.e02 = this->e02 / b.e02;

		s.e10 = this->e10 / b.e10;
		s.e11 = this->e11 / b.e11;
		s.e12 = this->e12 / b.e12;

		s.e20 = this->e20 / b.e20;
		s.e21 = this->e21 / b.e21;
		s.e22 = this->e22 / b.e22;
		return s;
	}


};
#endif
