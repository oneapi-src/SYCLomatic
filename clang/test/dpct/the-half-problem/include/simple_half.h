#ifndef _HALF_H_
#define _HALF_H_
class half
{
    unsigned short _h;
  public:
    //-------------
    // Constructors
    //-------------
    half ();			// no initialization
    half (float f);
    unsigned short	bits () const;
    void		setBits (unsigned short bits);
};

inline
half::half () {}

inline
half::half (float f) { }

inline unsigned short
half::bits () const {
    return _h;
}

inline void
half::setBits (unsigned short bits) {
    _h = bits;
}
#undef HALF_EXPORT_CONST
#endif
