// Exact payload rows of an IVF index's stored documents (#23), recovered from its fast list storage
// (tieredq_fast_rows.h), so rescoring tiers such as TieredQRefinement need no row-major copy of the codes.
#include "tieredq/tieredq.h"
#include "tieredq_fast_rows.h"

namespace dv {
using tieredq_detail::bad;
namespace { constexpr uint32_t padding_id=UINT32_MAX; }

std::vector<uint32_t> TieredQIVFIndex::id_documents() const {
  std::vector<uint32_t> documents(size_,padding_id);
  for (uint32_t list=0;list<lists();++list)
    ids_.for_each(list,[&](size_t position,uint32_t id) { documents[id]=uint32_t(list_begin_[list]+position); });
  return documents;
}

void TieredQIVFIndex::document_rows(const uint32_t* documents,size_t count,uint8_t* rows) const {
  if (!count) return;
  if (!documents || !rows) bad("null document buffer");
  const TieredQ& model=codec();
  if (documents_.fine_tiers_) bad("document_rows does not serve fine-tier models");
  for (size_t i=0;i<count;++i)
    if (!live(documents[i])) bad("invalid stored document");
  tieredq_detail::read_fast_rows(*documents_.fast_,model.levels_,model.tiers_,model.coded_dimensions(),model.code_bytes(),
                                 model.bytes_per_vector(),documents,count,rows);
}
} // namespace dv
