//! Per-document native export cache; file writes and save identity stay with the caller.

use std::sync::{Arc, Mutex};

use fotodit_doc::Document;
use fotodit_format::{PlyraWriter, SaveOptions};

/// Shared native export cache. Clones reuse tile hashes and up to 64 MiB of compressed payloads;
/// dropping the last clone releases it, including when a document closes after a save worker.
#[derive(Clone)]
pub struct PlyraExportCache(Arc<Mutex<PlyraWriter>>);

impl Default for PlyraExportCache {
    fn default() -> Self {
        let mut writer = PlyraWriter::new();
        writer.set_compressed_cache_budget(64 << 20);
        Self(Arc::new(Mutex::new(writer)))
    }
}

impl std::fmt::Debug for PlyraExportCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PlyraExportCache")
    }
}

impl PartialEq for PlyraExportCache {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PlyraExportCache {
    pub(crate) fn encode(&self, doc: &Document, opts: &SaveOptions) -> fotodit_format::Result<Vec<u8>> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).save_zip(doc, opts).map(|(bytes, _)| bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExportOptions, export};
    use fotodit_color::{ColorMode, SampleType};
    use fotodit_doc::{Layer, LayerContent};
    use fotodit_geom::Size;
    use fotodit_raster::Surface;

    #[test]
    fn cached_exports_reuse_tiles_and_preserve_edits_and_save_as() {
        for depth in SampleType::ALL {
            let mut doc = Document::new("first.plyra", Size::new(2, 1), ColorMode::Rgb, depth);
            let mut surface = Surface::new(doc.pixel_format());
            surface.write_pixel(0, 0, &[0.2, 0.4, 0.6, 1.0]);
            doc.layers.push(Layer::new("photo", LayerContent::Raster(surface)));
            let cache = PlyraExportCache::default();
            let options = ExportOptions { plyra_cache: Some(cache.clone()), ..Default::default() };
            let first = export(&doc, "first.plyra", &options).unwrap();
            assert_eq!(fotodit_format::load_from_bytes(&first.bytes).unwrap(), doc);
            let (_, stats) = cache.0.lock().unwrap().save_zip(&doc, &SaveOptions::default()).unwrap();
            assert_eq!(stats.tiles_reused, stats.tiles_total);
            assert_eq!(stats.tiles_written, 0);
            doc.layers[0].surface_mut().unwrap().try_write_region(fotodit_geom::Rect::new(0, 0, 1, 1), &[0.8, 0.3, 0.1, 1.0]).unwrap();
            let next = export(&doc, "second.plyra", &options).unwrap();
            let mut expected = doc.clone();
            expected.name = "second.plyra".into();
            assert_eq!(fotodit_format::load_from_bytes(&next.bytes).unwrap(), expected);
            assert_eq!(next.bytes, export(&doc, "second.plyra", &ExportOptions::default()).unwrap().bytes);
            let weak = Arc::downgrade(&cache.0);
            drop(options);
            drop(cache);
            assert!(weak.upgrade().is_none());
        }
    }
}
