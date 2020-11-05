#ifndef DEBUGPRINT_H_
#define DEBUGPRINT_H_

#include <stdio.h>
#include <string.h>

#define DEBUG_TAG

namespace DeepLearning {

// error information will always be displayed.
#define DBG_ERROR(a) do { \
   const char *pFileName = strrchr(__FILE__, '/'); \
   if (!pFileName) \
   { \
      pFileName = __FILE__; \
   } \
   else \
   { \
     pFileName++; \
   } \
   printf("Error: %s: %s(): line %d: ", pFileName, __FUNCTION__, __LINE__); \
   printf a; \
   printf("\n"); \
} while(0)

#ifdef DEBUG_TAG

#define DBG_MSG(a) do { \
   const char *pFileName = strrchr(__FILE__, '/'); \
   if (!pFileName) \
   { \
      pFileName = __FILE__; \
   } \
   else \
   { \
     pFileName++; \
   } \
   printf("MSG  : %s: %s(): line %d: ", pFileName, __FUNCTION__, __LINE__); \
   printf a; \
   printf("\n"); \
} while(0)

#else

#define DBG_MSG(a)

#endif /* DEBUG_TAG */

} /* namespace DeepLearning */

#endif /* DEBUGPRINT_H_ */
