package io.github.nierrecomp;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ShortcutInfo;
import android.content.pm.ShortcutManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.Icon;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The home screen in front of the game (a NativeActivity, libnier.so): checks for the player's data in Internal
 * storage/NieR Automata (asking for all-files access), shows the options and writes them to that folder's config.txt,
 * whose NX_* lines the runtime turns into its switches (the other keys it reads as settings):
 *   game text   the language of the game's texts (language=auto, default: the device's; or en-US, es, fr, de, it, ja,
 *               ko, zh-Hans, zh-Hant: runtime language.h hands it to the game as the console language)
 *   resolution  360p, 480p, 720p (default) or 1080p: NX_HANDHELD (1 below 1080p), resolution, NX_RENDER_RES, NX_RES
 *   fullscreen  the picture stretched to the whole screen (NX_STRETCH=1) or kept 16:9 (0, default)
 *   graphics    low (default), medium or high: textures one mip smaller (NX_LOD_BIAS=1), half a mip, or the game's own
 *   frame rate  30 (NX_FPS_CAP=30, default: runtime nvn_queue.cpp holds the present interval at 2) or 60
 *   shadows     on (NX_SHADOWS=1, default) or off (0: runtime nvn_draw.cpp skips the draws into shadow maps)
 *   minimum dynamic resolution  NX_NIER_MIN_SCALE: how low the game's dynamic resolution may take its 3D when the GPU
 *               is behind, 0.45, 0.5 (default), 0.6, 0.7 or 0.8333 (the game's own floor; games/nier/game.cpp)
 *   bloom       experimental: on (NX_DISABLE_BLOOM=0, default) or off (1: runtime nvn_draw.cpp skips every draw into a
 *               colour target's smaller mip levels, which is where bloom and blur chains go, and maybe other effects)
 *   FPS counter off (default) or on (NX_SHOW_FPS)
 *   on-screen controls on (default) or off (NX_TOUCH_SHOW), their size (NX_TOUCH_SIZE) and opacity (NX_TOUCH_OPACITY),
 *               and where they are (NX_TOUCH_LAYOUT, written by ControlsEditor)
 *   touch stick and camera sensitivity  NX_STICK_SENSITIVITY and NX_CAMERA_SENSITIVITY, percent (100 default): how far
 *               the drawn left stick and the camera drag go for the same thumb travel (platform_android_input.cpp)
 *   gamepad     physical controller buttons by position, Nintendo (NX_GAMEPAD_LAYOUT=0, default: the bottom button is
 *               B) or by label, Xbox / PC (1: the bottom button is A)
 * It imports or exports the saves (user/save, SaveFiles.java: a ZIP or a folder in, a ZIP in Download out), installs
 * or turns on and off the mods (mods/<Name>/romfs, Mods.java) and exports the logs (logcat tag "nier" and nx.log) to
 * Download/nier-log.txt.
 * The game data: the player picks their base game (.xci / .nsp), update 1.0.3 (.nsp) and keys here, and
 * ImportActivity writes the game with its update to Internal storage/NieR Automata (exefs/, extract/romfs/) and
 * translates its shaders (spirv/), once. A data folder prepared on the PC (the same layout as private/nier) works too.
 * The icon: a home screen shortcut with a picture of the player's own (Android lets an app pin shortcuts, not change
 * its own icon).
 * Other lines of config.txt are kept as they were. The launcher itself speaks the device's language (Lang.java:
 * Spanish, English, French, German, Italian, Portuguese, Japanese, Korean, Chinese; English otherwise).
 */
public class Launcher extends Activity {
    static final String DATA = "NieR Automata";

    // The launcher's colours: dark warm grey, the menus' parchment and their dark ink.
    static final int BACKGROUND = Color.rgb(24, 23, 20), PARCHMENT = Color.rgb(218, 212, 187),
            INK = Color.rgb(52, 49, 42);

    static final class Option {
        final String key, es, en;
        final String[] values, labelsEs, labelsEn;
        int pick;

        Option(String key, String es, String en, String[] values, String[] labelsEs, String[] labelsEn, int pick) {
            this.key = key;
            this.es = es;
            this.en = en;
            this.values = values;
            this.labelsEs = labelsEs;
            this.labelsEn = labelsEn;
            this.pick = pick;
        }
    }

    final List<Option> options = new ArrayList<>();
    final Map<String, String> config = new LinkedHashMap<>();
    TextView status;
    Button play;
    TextView driverName;
    static final int PICK_ZIP = 1, PICK_FOLDER = 2, PICK_ICON = 3, PICK_MOD = 4, PICK_DRIVER = 5, PICK_BASE = 6, PICK_UPDATE = 7,
            PICK_KEYS = 8, PREPARE = 9;

    String t(String spanish, String english) { return Lang.t(spanish, english); }

    String label(Option o) { return t(o.labelsEs[o.pick], o.labelsEn[o.pick]); }

    static File dataDir() { return new File(Environment.getExternalStorageDirectory(), DATA); }

    // Android 10 (API 29, the minimum) has no Environment.isExternalStorageManager: the call sits in a class of its own,
    // loaded only on 11 and later, so verifying the launcher on 10 never meets the missing method.
    static final class Api30Helper {
        static boolean isExternalStorageManager() { return Environment.isExternalStorageManager(); }
    }

    boolean hasAccess() {
        if (Build.VERSION.SDK_INT < 30) return true;
        return Api30Helper.isExternalStorageManager();
    }

    String displayName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Exception ignored) {
        }
        String last = uri.getLastPathSegment();
        return last == null ? "" : last.substring(last.lastIndexOf('/') + 1);
    }

    // ---- the game, prepared on the phone: the base game and update 1.0.3 the player picks, with their keys ----

    SharedPreferences prefs() { return getSharedPreferences("game", MODE_PRIVATE); }

    String gameUri(boolean update) { return prefs().getString(update ? "update_uri" : "base_uri", null); }

    String gameName(boolean update) { return prefs().getString(update ? "update_name" : "base_name", ""); }

    static File keysFile(Context c, String name) { return new File(new File(c.getFilesDir(), "keys"), name); }

    boolean hasKeys() { return keysFile(this, "prod.keys").length() > 0; }

    boolean readable(String uri) {
        if (uri == null) return false;
        try (ParcelFileDescriptor f = getContentResolver().openFileDescriptor(Uri.parse(uri), "r")) {
            return f != null;
        } catch (Exception e) {
            return false;
        }
    }

    // The base game (.xci / .nsp) or the update (.nsp); the update is small (about 24 MB), the base game is not.
    void onGamePicked(Uri uri, boolean update) {
        String name = displayName(uri);
        String lower = name.toLowerCase(Locale.ROOT);
        long size = -1;
        try (ParcelFileDescriptor f = getContentResolver().openFileDescriptor(uri, "r")) {
            if (f != null) size = f.getStatSize();
        } catch (Exception ignored) {
        }
        boolean game = lower.endsWith(".nsp") || lower.endsWith(".xci");
        boolean looksUpdate = lower.endsWith(".nsp") && size >= 0 && size < (1L << 30);
        if (!game || (update && !lower.endsWith(".nsp")) || (update != looksUpdate && size >= 0)) {
            tell(t("Archivo no válido", "Not the right file"), update
                    ? t("Elige el .nsp de la actualización 1.0.3 (v131072, unos 24 MB).",
                            "Choose update 1.0.3's .nsp (v131072, about 24 MB).")
                    : t("Elige el .xci o .nsp del juego base (el grande, no la actualización).",
                            "Choose the base game's .xci or .nsp (the big one, not the update)."));
            return;
        }
        try {
            getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (Exception ignored) {  // a provider without persistable grants: works while the app runs
        }
        prefs().edit().putString(update ? "update_uri" : "base_uri", uri.toString())
                .putString(update ? "update_name" : "base_name", name).apply();
    }

    // prod.keys (and title.keys, when the dump needs it): copied into the app's own storage, never shared or logged.
    void onKeysPicked(Intent data) {
        List<Uri> uris = new ArrayList<>();
        ClipData clip = data.getClipData();
        if (clip != null)
            for (int k = 0; k < clip.getItemCount(); k++) uris.add(clip.getItemAt(k).getUri());
        else if (data.getData() != null) uris.add(data.getData());
        boolean prod = false;
        for (Uri uri : uris) {
            String name = displayName(uri).toLowerCase(Locale.ROOT);
            String target = name.contains("title") ? "title.keys" : "prod.keys";
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                byte[] buf = new byte[1 << 20];
                int n = 0, r;
                while (in != null && n < buf.length && (r = in.read(buf, n, buf.length - n)) > 0) n += r;
                String text = new String(buf, 0, n, "UTF-8");
                if (target.equals("prod.keys") && !text.contains("header_key") && !text.contains("master_key_00")) continue;
                File f = keysFile(this, target);
                f.getParentFile().mkdirs();
                try (OutputStream out = new FileOutputStream(f)) {
                    out.write(buf, 0, n);
                }
                prod |= target.equals("prod.keys");
            } catch (Exception ignored) {
            }
        }
        if (!prod && !hasKeys())
            tell(t("Claves no válidas", "Not a keys file"),
                    t("Ese archivo no tiene las claves (header_key). Elige el prod.keys de tu consola.",
                            "That file has no keys (header_key). Choose your console's prod.keys."));
    }

    void prepare() {
        Intent i = new Intent(this, ImportActivity.class);
        i.putExtra(ImportActivity.EXTRA_BASE, gameUri(false));
        i.putExtra(ImportActivity.EXTRA_BASE_NAME, gameName(false));
        i.putExtra(ImportActivity.EXTRA_UPDATE, gameUri(true));
        i.putExtra(ImportActivity.EXTRA_UPDATE_NAME, gameName(true));
        startActivityForResult(i, PREPARE);
    }

    boolean dataReady() { return missingData().isEmpty() && wrongBuild() == null; }

    // What the play button does first when the data is not ready: pick what is missing, then prepare.
    void nextStep() {
        if (!readable(gameUri(false))) pick(PICK_BASE, "*/*", false);
        else if (!readable(gameUri(true))) pick(PICK_UPDATE, "*/*", false);
        else if (!hasKeys()) pick(PICK_KEYS, "*/*", true);
        else prepare();
    }

    void pick(int request, String type, boolean multiple) {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType(type);
        if (multiple) i.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        startActivityForResult(i, request);
    }

    // A home screen shortcut with the player's picture (cropped to a square; an adaptive icon, so the home screen gives
    // it its own shape).
    void onIconPicked(Uri uri) {
        try {
            Bitmap src;
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                src = BitmapFactory.decodeStream(in);
            }
            if (src == null) throw new IllegalArgumentException("not a picture");
            int side = Math.min(src.getWidth(), src.getHeight());
            Bitmap square = Bitmap.createBitmap(src, (src.getWidth() - side) / 2, (src.getHeight() - side) / 2, side, side);
            Bitmap icon = Bitmap.createScaledBitmap(square, 432, 432, true);
            ShortcutManager sm = getSystemService(ShortcutManager.class);
            if (sm == null || !sm.isRequestPinShortcutSupported()) {
                tell(t("Icono", "Icon"), t("Tu pantalla de inicio no admite accesos directos.",
                        "Your home screen does not support shortcuts."));
                return;
            }
            Intent launch = new Intent(this, Launcher.class).setAction(Intent.ACTION_MAIN);
            ShortcutInfo info = new ShortcutInfo.Builder(this, "picture_" + System.currentTimeMillis())
                    .setShortLabel("NieR").setLongLabel("NieR Automata").setIcon(Icon.createWithAdaptiveBitmap(icon))
                    .setIntent(launch).build();
            sm.requestPinShortcut(info, null);
            tell(t("Icono", "Icon"), t("Acepta el aviso para poner el acceso directo con tu imagen en la pantalla de inicio. "
                            + "El icono del cajón de apps no cambia: Android no deja que una app cambie el suyo.",
                    "Accept the prompt to put the shortcut with your picture on the home screen. The icon in the app "
                            + "drawer stays: Android does not let an app change its own."));
        } catch (Exception e) {
            tell(t("Icono", "Icon"), t("No se pudo usar esa imagen.", "That picture could not be used."));
        }
    }

    // ---- the game data, prepared on the PC ----

    // What the game needs in the data folder: the game's files (extract/romfs, with its CPKs), the translated shaders
    // (spirv/, results.tsv holds their records) and the game's code: analysis/main.bin for the recompiled build (the
    // runtime's loader.cpp), exefs/main for the native one.
    List<String> missingData() {
        List<String> missing = new ArrayList<>();
        File d = dataDir();
        if (!new File(d, "extract/romfs/data000.cpk").canRead()) missing.add("extract/romfs/data000.cpk");
        if (!new File(d, "spirv/results.tsv").canRead()) missing.add("spirv/results.tsv");
        if (!new File(d, "exefs/main").canRead()) missing.add("exefs/main");
        if (!new File(d, "exefs/sdk").canRead()) missing.add("exefs/sdk");
        return missing;
    }

    // The build ID the port is made for (NieR:Automata The End of YoRHa Edition, update 1.0.3): the start of
    // exefs/main's module ID (NSO header +0x40), as games/nier/game.cpp names it. The native build refuses another
    // (runtime main.cpp); say so here instead of closing.
    static final String BUILD_ID = "3C10EC35C5EF4C36";

    // Null when exefs/main is missing or the right version, else its build ID.
    String wrongBuild() {
        File main = new File(dataDir(), "exefs/main");
        if (!main.canRead()) return null;
        byte[] id = new byte[8];
        try (java.io.RandomAccessFile f = new java.io.RandomAccessFile(main, "r")) {
            f.seek(0x40);
            f.readFully(id);
        } catch (java.io.IOException e) {
            return "?";
        }
        StringBuilder hex = new StringBuilder();
        for (byte b : id) hex.append(String.format("%02X", b & 0xFF));
        return hex.toString().equals(BUILD_ID) ? null : hex.toString();
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        // The game's texts (runtime language.h): its own language names, as a console lists them.
        options.add(new Option("language", "Idioma de textos y subtítulos", "Text and subtitle language",
                new String[] {"auto", "en-US", "es", "fr", "de", "it", "ja", "ko", "zh-Hans", "zh-Hant"},
                new String[] {"Automático (idioma del dispositivo)", "English", "Español", "Français", "Deutsch",
                        "Italiano", "日本語", "한국어", "简体中文", "繁體中文"},
                new String[] {"Automatic (device language)", "English", "Español", "Français", "Deutsch", "Italiano",
                        "日本語", "한국어", "简体中文", "繁體中文"}, 0));
        options.add(new Option("RESOLUTION", "Resolución", "Resolution",
                new String[] {"360p", "480p", "720p", "1080p"},
                new String[] {"360p", "480p", "720p", "1080p"},
                new String[] {"360p", "480p", "720p", "1080p"}, 2));
        options.add(new Option("NX_STRETCH", "Pantalla completa (estirar)", "Fullscreen (stretch)",
                new String[] {"0", "1"},
                new String[] {"Desactivada", "Activada"},
                new String[] {"Off", "On"}, 0));
        options.add(new Option("GRAPHICS", "Gráficos", "Graphics", new String[] {"low", "medium", "high"},
                new String[] {"Bajos", "Medios", "Altos"}, new String[] {"Low", "Medium", "High"}, 0));
        options.add(new Option("NX_FPS_CAP", "Fotogramas por segundo", "Frame rate", new String[] {"30", "60"},
                new String[] {"30 fps", "60 fps (experimental)"}, new String[] {"30 fps", "60 fps (experimental)"}, 0));
        options.add(new Option("NX_SHADOWS", "Sombras", "Shadows", new String[] {"0", "1"},
                new String[] {"Desactivadas", "Activadas"}, new String[] {"Off", "On"}, 1));
        options.add(new Option("NX_NIER_MIN_SCALE", "Resolución dinámica mínima", "Minimum dynamic resolution",
                new String[] {"0.5", "0.6", "0.7", "0.8333"},
                new String[] {"50 % (más fluido)", "60 %", "70 %", "83 % (la del juego)"},
                new String[] {"50% (smoother)", "60%", "70%", "83% (the game's)"}, 1));
        // Experimental: the runtime's heuristic (draws into mip levels above 0) was made on P5R, so it may take other
        // effects with the bloom. Bloom stays on unless the player turns it off.
        options.add(new Option("NX_DISABLE_BLOOM", "Bloom (experimental)", "Bloom (experimental)",
                new String[] {"0", "1"},
                new String[] {"Activado (normal)", "Desactivado"},
                new String[] {"On (normal)", "Off"}, 0));
        options.add(new Option("NX_SHOW_FPS", "Contador de FPS", "FPS counter", new String[] {"0", "1"},
                new String[] {"Desactivado", "Activado"}, new String[] {"Off", "On"}, 0));
        options.add(new Option("NX_TOUCH_SHOW", "Controles táctiles", "On-screen controls", new String[] {"1", "0"},
                new String[] {"Activados", "Desactivados"}, new String[] {"On", "Off"}, 0));
        options.add(new Option("NX_TOUCH_SIZE", "Tamaño de los botones", "Button size",
                new String[] {"80", "100", "125"}, new String[] {"Pequeño", "Normal", "Grande"},
                new String[] {"Small", "Normal", "Large"}, 1));
        options.add(new Option("NX_TOUCH_OPACITY", "Opacidad de los botones", "Button opacity",
                new String[] {"50", "85", "100"}, new String[] {"Baja", "Normal", "Alta"},
                new String[] {"Low", "Normal", "High"}, 1));
        options.add(new Option("NX_STICK_SENSITIVITY", "Sensibilidad de la palanca táctil", "Touch stick sensitivity",
                new String[] {"50", "75", "100", "125", "150", "200"},
                new String[] {"50%", "75%", "100%", "125%", "150%", "200%"},
                new String[] {"50%", "75%", "100%", "125%", "150%", "200%"}, 2));
        options.add(new Option("NX_CAMERA_SENSITIVITY", "Sensibilidad de la cámara táctil", "Touch camera sensitivity",
                new String[] {"50", "75", "100", "125", "150", "200", "250"},
                new String[] {"50%", "75%", "100%", "125%", "150%", "200%", "250%"},
                new String[] {"50%", "75%", "100%", "125%", "150%", "200%", "250%"}, 2));
        options.add(new Option("NX_GAMEPAD_LAYOUT", "Botones del mando físico", "Physical gamepad buttons",
                new String[] {"0", "1"},
                new String[] {"Nintendo (A a la derecha, B abajo)", "Xbox / PC (A abajo, B a la derecha)"},
                new String[] {"Nintendo (A right, B down)", "Xbox / PC (A down, B right)"}, 0));
        build();
    }

    @Override
    protected void onResume() {
        super.onResume();
        refresh();
    }

    void build() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BACKGROUND);
        int pad = dp(24);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("NieR:Automata");
        title.setTextColor(PARCHMENT);
        title.setTextSize(30);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);

        status = new TextView(this);
        status.setTextColor(Color.rgb(220, 220, 220));
        status.setTextSize(14);
        status.setPadding(0, dp(6), 0, dp(14));
        root.addView(status);

        addRow(root, t("Juego base (.xci / .nsp)", "Base game (.xci / .nsp)"), t("Elegir", "Choose"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pick(PICK_BASE, "*/*", false);
            }
        });
        addRow(root, t("Actualización 1.0.3 (.nsp)", "Update 1.0.3 (.nsp)"), t("Elegir", "Choose"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pick(PICK_UPDATE, "*/*", false);
            }
        });
        addRow(root, t("Claves (prod.keys)", "Keys (prod.keys)"), t("Elegir", "Choose"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pick(PICK_KEYS, "*/*", true);
            }
        });

        addRow(root, t("Icono", "Icon"), t("Elegir imagen", "Choose a picture"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pick(PICK_ICON, "image/*", false);
            }
        });

        for (final Option o : options) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(0, dp(4), 0, dp(4));
            TextView name = new TextView(this);
            name.setText(t(o.es, o.en));
            name.setTextColor(Color.WHITE);
            name.setTextSize(17);
            row.addView(name, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
            final Button value = new Button(this);
            value.setAllCaps(false);
            value.setMinWidth(dp(180));
            row.addView(value);
            value.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    o.pick = (o.pick + 1) % o.values.length;
                    value.setText(label(o));
                    save();
                }
            });
            value.setTag(o);
            root.addView(row);
        }

        addRow(root, t("Posición de los controles", "Controls position"), t("Mover", "Move"),
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if (!hasAccess()) {
                            askAccess();
                            return;
                        }
                        save();
                        startActivity(new Intent(Launcher.this, ControlsEditor.class));
                    }
                });

        // Saves: in from a ZIP or a folder (a Switch backup or another player's), out to a ZIP in Download.
        addPairRow(root, t("Partidas guardadas", "Saves"), t("Importar", "Import"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!hasAccess()) {
                    askAccess();
                    return;
                }
                askImport();
            }
        }, t("Exportar", "Export"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!hasAccess()) {
                    askAccess();
                    return;
                }
                exportSaves();
            }
        });

        // Mods: a ZIP (its romfs/ folder) or a .cpk in, and each mod in mods/ turned on or off.
        addPairRow(root, t("Mods", "Mods"), t("Instalar", "Install"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!hasAccess()) {
                    askAccess();
                    return;
                }
                pick(PICK_MOD, "*/*", false);
            }
        }, t("Gestionar", "Manage"), new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!hasAccess()) {
                    askAccess();
                    return;
                }
                manageMods();
            }
        });

        // GPU driver (libadrenotools / Turnip)
        LinearLayout drvRow = new LinearLayout(this);
        drvRow.setOrientation(LinearLayout.HORIZONTAL);
        drvRow.setGravity(Gravity.CENTER_VERTICAL);
        drvRow.setPadding(0, dp(4), 0, dp(4));
        driverName = new TextView(this);
        driverName.setTextColor(Color.WHITE);
        drvRow.addView(driverName, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        Button drvImport = new Button(this);
        drvImport.setAllCaps(false);
        drvImport.setMinWidth(dp(88));
        drvImport.setText(t("Importar", "Import"));
        drvRow.addView(drvImport);
        Button drvRemove = new Button(this);
        drvRemove.setAllCaps(false);
        drvRemove.setMinWidth(dp(88));
        drvRemove.setText(t("Quitar", "Remove"));
        drvRow.addView(drvRemove);
        drvImport.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pick(PICK_DRIVER, "*/*", false);
            }
        });
        drvRemove.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                removeDriver();
            }
        });
        root.addView(drvRow);
        showDriver();

        // Logs: logcat (the runtime's tag is the game id) and nx.log, into one text file in Download to send.
        addRow(root, t("Registros de depuración", "Debug logs"), t("Exportar registros", "Export logs"),
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if (!hasAccess()) {
                            askAccess();
                            return;
                        }
                        exportLogs();
                    }
                });

        play = new Button(this);
        play.setTextSize(20);
        play.setAllCaps(false);
        play.setBackgroundColor(PARCHMENT);
        play.setTextColor(INK);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(64));
        lp.topMargin = dp(18);
        root.addView(play, lp);
        play.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!hasAccess()) {
                    askAccess();
                    return;
                }
                if (!dataReady()) {
                    nextStep();
                    return;
                }
                save();
                startActivity(new Intent(Launcher.this, android.app.NativeActivity.class));
            }
        });

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);
    }

    void refresh() {
        load();
        for (Option o : options) {
            String v = config.get(o.key);
            if ("RESOLUTION".equals(o.key) && v == null) {
                String hh = config.get("NX_HANDHELD");
                if ("0".equals(hh)) v = "1080p";
                else {
                    String r = config.get("resolution");
                    if (r != null && r.contains("360")) v = "360p";
                    else if (r != null && r.contains("480")) v = "480p";
                    else v = "720p";
                }
            }
            for (int k = 0; v != null && k < o.values.length; k++)
                if (o.values[k].equals(v)) o.pick = k;
        }
        updateButtons(getWindow().getDecorView());
        showDriver();
        List<String> missing = hasAccess() ? missingData() : new ArrayList<String>();
        if (!hasAccess()) {
            status.setText(t("Hace falta el permiso de acceso a todos los archivos para leer los datos del juego.",
                    "All-files access is needed to read the game data."));
            play.setText(t("Dar permiso", "Grant access"));
        } else if (!dataReady() && !readable(gameUri(false))) {
            status.setText(t("Elige tu juego base de NieR:Automata The End of YoRHa Edition (.xci o .nsp, juego base de "
                            + "EE. UU.), luego la actualización 1.0.3 y tus claves. La app prepara el juego una sola vez "
                            + "(unos 12 GB en el almacenamiento).",
                    "Choose your NieR:Automata The End of YoRHa Edition base game (.xci or .nsp, the US base game), then "
                            + "update 1.0.3 and your keys. The app prepares the game once (about 12 GB of storage)."));
            play.setText(t("Elegir el juego base", "Choose the base game"));
        } else if (!dataReady() && !readable(gameUri(true))) {
            status.setText(Lang.f("Juego base: %s. Falta la actualización 1.0.3 (.nsp, v131072).",
                    "Base game: %s. Update 1.0.3 (.nsp, v131072) is missing.", gameName(false)));
            play.setText(t("Elegir la actualización 1.0.3", "Choose update 1.0.3"));
        } else if (!dataReady() && !hasKeys()) {
            status.setText(t("Falta tu prod.keys (y title.keys si tu volcado lo necesita).",
                    "Your prod.keys (and title.keys, if your dump needs it) is missing."));
            play.setText(t("Elegir prod.keys", "Choose prod.keys"));
        } else if (!dataReady()) {
            status.setText(Lang.f("Listo para preparar: %s + %s. Copia el juego con su actualización a Almacenamiento "
                            + "interno/%s y traduce sus shaders, una sola vez.",
                    "Ready to prepare: %s + %s. It writes the game with its update to Internal storage/%s and translates "
                            + "its shaders, once.", gameName(false), gameName(true), DATA));
            play.setText(t("Preparar el juego", "Prepare the game"));
        } else {
            status.setText(Lang.f("Datos listos: %s", "Data ready: %s", dataDir().getPath()));
            play.setText(t("Jugar", "Play"));
        }
    }

    void updateButtons(View v) {
        if (v instanceof Button && v.getTag() instanceof Option) {
            ((Button) v).setText(label((Option) v.getTag()));
        } else if (v instanceof android.view.ViewGroup) {
            android.view.ViewGroup g = (android.view.ViewGroup) v;
            for (int k = 0; k < g.getChildCount(); k++) updateButtons(g.getChildAt(k));
        }
    }

    void tell(String title, String message) {
        new AlertDialog.Builder(this).setTitle(title).setMessage(message).setPositiveButton("OK", null).show();
    }

    // ---- saves ----

    void askImport() {
        new AlertDialog.Builder(this)
                .setTitle(t("Importar partidas", "Import saves"))
                .setMessage(t("Elige un archivo ZIP o una carpeta con partidas de NieR Automata (los archivos GameData y "
                                + "SlotData_0, SlotData_1..., como los exporta JKSV o Checkpoint, o el ZIP que exporta "
                                + "este botón). Antes se guarda una copia de tus partidas actuales.",
                        "Pick a ZIP file or a folder with NieR Automata saves (the GameData and SlotData_0, SlotData_1... "
                                + "files, as JKSV or Checkpoint export them, or the ZIP the Export button makes). Your "
                                + "current saves are copied aside first."))
                .setPositiveButton(t("Archivo ZIP", "ZIP file"), new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface d, int which) {
                        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                        i.addCategory(Intent.CATEGORY_OPENABLE);
                        i.setType("*/*");
                        i.putExtra(Intent.EXTRA_MIME_TYPES, new String[] {"application/zip",
                                "application/x-zip-compressed", "application/octet-stream"});
                        startActivityForResult(i, PICK_ZIP);
                    }
                })
                .setNeutralButton(t("Carpeta", "Folder"), new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface d, int which) {
                        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK_FOLDER);
                    }
                })
                .setNegativeButton(t("Cancelar", "Cancel"), null)
                .show();
    }

    void importSaves(int request, Uri uri) {
        try {
            SaveFiles.Result r = request == PICK_ZIP ? SaveFiles.importZip(getContentResolver(), uri)
                    : SaveFiles.importTree(getContentResolver(), uri);
            if (r == null) {
                tell(t("No hay partidas", "No saves found"),
                        t("Ahí no encontré partidas de NieR Automata: busca los archivos GameData o SlotData_0, "
                                        + "SlotData_1...",
                                "No NieR Automata saves there: look for the GameData or SlotData_0, SlotData_1... files."));
                return;
            }
            String kept = r.backup == null ? "" : Lang.f("\n\nTus partidas anteriores quedaron en user/%s.",
                    "\n\nYour previous saves are in user/%s.", r.backup);
            tell(t("Partidas importadas", "Saves imported"),
                    Lang.f("%d archivos (%s). El juego los carga en su próximo inicio.",
                            "%d files (%s). The game loads them at its next start.", r.files,
                            android.text.TextUtils.join(", ", r.saves)) + kept);
        } catch (Exception e) {
            tell(t("No se pudo importar", "Import failed"), String.valueOf(e.getMessage()));
        }
    }

    void exportSaves() {
        try {
            java.io.File out = SaveFiles.export();
            // Listed at once by the file pickers and the Files app (written straight to disk, it would wait for a scan).
            if (out != null) android.media.MediaScannerConnection.scanFile(this, new String[] {out.getPath()}, null, null);
            if (out == null) tell(t("No hay partidas", "No saves"), t("Todavía no hay partidas guardadas.", "There are no saves yet."));
            else tell(t("Partidas exportadas", "Saves exported"),
                    Lang.f("Guardadas en Descargas/%s. Se importan tal cual con el botón Importar.",
                            "Saved to Download/%s. The Import button takes it back as is.", out.getName()));
        } catch (Exception e) {
            tell(t("No se pudo exportar", "Export failed"), String.valueOf(e.getMessage()));
        }
    }

    // ---- mods ----

    // A ZIP or a .cpk, copied in on a thread of its own (a mod can be large) behind a dialog that waits for it.
    void installMod(final Uri uri) {
        final String name = displayName(uri);
        final String lower = name.toLowerCase(Locale.ROOT);
        if (!lower.endsWith(".zip") && !lower.endsWith(".cpk")) {
            tell(t("No es un mod", "Not a mod"), t("Elige el archivo ZIP de un mod (con una carpeta romfs dentro) o un "
                            + "archivo .cpk que reemplace uno del juego.",
                    "Choose a mod's ZIP file (with a romfs folder inside) or a .cpk file that replaces one of the game's."));
            return;
        }
        final AlertDialog wait = new AlertDialog.Builder(this).setTitle(t("Mods", "Mods"))
                .setMessage(t("Instalando el mod…", "Installing the mod…")).setCancelable(false).show();
        new Thread(new Runnable() {
            @Override
            public void run() {
                Mods.Result r = null;
                String error = null;
                try {
                    r = lower.endsWith(".zip") ? Mods.installZip(getContentResolver(), uri, name)
                            : Mods.installCpk(getContentResolver(), uri, name);
                } catch (Exception e) {
                    error = String.valueOf(e.getMessage());
                }
                final Mods.Result done = r;
                final String failed = error;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (isDestroyed()) return;  // the home screen closed while the mod was copied
                        wait.dismiss();
                        if (failed != null) tell(t("No se pudo instalar", "Install failed"), failed);
                        else if (done == null)
                            tell(t("No es un mod", "Not a mod"), t("Ese ZIP no tiene una carpeta romfs. Un mod lleva los "
                                            + "archivos que reemplaza en romfs/, con las mismas rutas que los del juego.",
                                    "That ZIP has no romfs folder. A mod holds the files it replaces under romfs/, with "
                                            + "the same paths as the game's."));
                        else tell(t("Mod instalado", "Mod installed"),
                                    Lang.f("%d archivos en mods/%s. El juego lo carga en su próximo inicio.",
                                            "%d files in mods/%s. The game loads it at its next start.", done.files,
                                            done.folder));
                    }
                });
            }
        }).start();
    }

    // The mods in mods/, each with a check box: on (Name) or off (_Name, which the runtime skips).
    void manageMods() {
        final List<Mods.Mod> mods = Mods.list();
        if (mods.isEmpty()) {
            tell(t("No hay mods", "No mods"), Lang.f("Todavía no hay mods. Instala un ZIP o un .cpk, o copia cada mod a "
                            + "Almacenamiento interno/%s/mods/<nombre>/romfs/ (con las mismas rutas que los archivos del "
                            + "juego).",
                    "No mods yet. Install a ZIP or a .cpk, or copy each mod to Internal storage/%s/mods/<name>/romfs/ "
                            + "(the same paths as the game's files).", DATA));
            return;
        }
        String[] names = new String[mods.size()];
        final boolean[] on = new boolean[mods.size()];
        for (int k = 0; k < mods.size(); k++) {
            names[k] = mods.get(k).name;
            on[k] = mods.get(k).on;
        }
        new AlertDialog.Builder(this)
                .setTitle(t("Mods activos", "Active mods"))
                .setMultiChoiceItems(names, on, new DialogInterface.OnMultiChoiceClickListener() {
                    @Override
                    public void onClick(DialogInterface d, int which, boolean checked) {
                        on[which] = checked;
                    }
                })
                .setPositiveButton(t("Aplicar", "Apply"), new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface d, int which) {
                        List<String> failed = new ArrayList<>();
                        for (int k = 0; k < mods.size(); k++)
                            if (!Mods.setOn(mods.get(k), on[k])) failed.add(mods.get(k).name);
                        if (!failed.isEmpty())
                            tell(t("Mods", "Mods"), Lang.f("No se pudo renombrar la carpeta de un mod: %s",
                                    "A mod folder could not be renamed: %s", android.text.TextUtils.join(", ", failed)));
                    }
                })
                .setNegativeButton(t("Cancelar", "Cancel"), null)
                .show();
    }

    // ---- logs ----

    static final int LOG_TAIL = 1 << 20;  // the end of nx.log that goes in (the file grows with every run)
    static final int SHARE_TEXT = 100000;  // what the share sheet gets as text (an Intent travels in ~1 MB at most)

    // The end of a text file, at most `max` bytes, from the start of a line.
    static String tail(File f, int max) throws java.io.IOException {
        try (java.io.RandomAccessFile r = new java.io.RandomAccessFile(f, "r")) {
            long start = Math.max(0, r.length() - max);
            byte[] buf = new byte[(int) (r.length() - start)];
            r.seek(start);
            r.readFully(buf);
            String s = new String(buf, "UTF-8");
            int nl = start > 0 ? s.indexOf('\n') : -1;
            return nl >= 0 ? s.substring(nl + 1) : s;
        }
    }

    // Download/nier-log.txt: the device, the runtime's logcat lines (its tag is the game id, "nier"), crash dumps
    // (DEBUG), the Vulkan driver's lines and the end of nx.log (written next to the data by non-release builds); then
    // the share sheet with the same text (its end, when long).
    File driverDir() {
        File dir = new File(getFilesDir(), "gpu_driver");
        dir.mkdirs();
        return dir;
    }

    String activeDriver() {
        File active = new File(driverDir(), "active.txt");
        if (!active.exists()) return "";
        try (BufferedReader r = new BufferedReader(new FileReader(active))) {
            String l = r.readLine();
            return l == null ? "" : l.trim();
        } catch (Exception e) {
            return "";
        }
    }

    void showDriver() {
        if (driverName == null) return;
        String a = activeDriver();
        driverName.setText(a.isEmpty() ? t("Driver de GPU: del sistema", "GPU driver: system")
                : String.format(Locale.ROOT, t("Driver de GPU: %s", "GPU driver: %s"), a));
    }

    void removeDriver() {
        File[] files = driverDir().listFiles();
        if (files != null) for (File f : files) f.delete();
        showDriver();
        Toast.makeText(this, t("Driver del sistema activado", "System driver restored"), Toast.LENGTH_LONG).show();
    }

    static void copyDriverStream(InputStream in, File dest) throws java.io.IOException {
        try (OutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[65536];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
    }

    void importDriver(Uri uri) {
        try {
            String picked = displayName(uri);
            File dir = driverDir();
            File[] old = dir.listFiles();
            if (old != null) for (File f : old) f.delete();
            String libName = null;
            if (picked.toLowerCase(Locale.ROOT).endsWith(".zip")) {
                String soEntry = null, wanted = null;
                try (java.util.zip.ZipInputStream zin = new java.util.zip.ZipInputStream(getContentResolver().openInputStream(uri))) {
                    java.util.zip.ZipEntry e;
                    while ((e = zin.getNextEntry()) != null) {
                        String base = new File(e.getName()).getName();
                        if (e.isDirectory()) continue;
                        if (base.equals("meta.json")) {
                            java.io.ByteArrayOutputStream bo = new java.io.ByteArrayOutputStream();
                            byte[] buf = new byte[8192];
                            int n;
                            while ((n = zin.read(buf)) > 0) bo.write(buf, 0, n);
                            java.util.regex.Matcher m = java.util.regex.Pattern.compile("\"libraryName\"\\s*:\\s*\"([^\"]+)\"")
                                    .matcher(bo.toString("UTF-8"));
                            if (m.find()) wanted = m.group(1);
                        } else if (base.endsWith(".so")) {
                            copyDriverStream(zin, new File(dir, base));
                            if (soEntry == null) soEntry = base;
                        }
                    }
                }
                libName = wanted != null && new File(dir, wanted).exists() ? wanted : soEntry;
            } else {
                libName = picked.endsWith(".so") ? picked : picked + ".so";
                try (InputStream in = getContentResolver().openInputStream(uri)) {
                    if (in == null) throw new java.io.IOException("Cannot open selected file");
                    copyDriverStream(in, new File(dir, libName));
                }
            }
            if (libName == null) throw new java.io.IOException(t("No hay ningun .so dentro", "No .so file inside"));
            try (FileWriter w = new FileWriter(new File(dir, "active.txt"))) {
                w.write(libName + "\n");
            }
            showDriver();
            new AlertDialog.Builder(this)
                    .setTitle(t("Driver importado", "Driver imported"))
                    .setMessage(String.format(Locale.ROOT, t("%s\n\nCierra el juego por completo y abrelo otra vez. Para volver al driver del sistema pulsa Quitar.",
                            "%s\n\nClose the game completely and start it again. To go back to the system driver press Remove."), libName))
                    .setPositiveButton("OK", null)
                    .show();
        } catch (Exception e) {
            new AlertDialog.Builder(this)
                    .setTitle(t("No se pudo importar el driver", "Driver import failed"))
                    .setMessage(String.valueOf(e.getMessage()))
                    .setPositiveButton("OK", null)
                    .show();
        }
    }

    void exportLogs() {
        try {
            File dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
            if (!dl.isDirectory() && !dl.mkdirs()) throw new java.io.IOException("cannot make " + dl);
            File out = new File(dl, "nier-log.txt");
            StringBuilder sb = new StringBuilder();
            sb.append("=== NieR Automata Recomp Log Export ===\n");
            try {
                sb.append("App: ").append(getPackageManager().getPackageInfo(getPackageName(), 0).versionName).append('\n');
            } catch (Exception ignored) {
            }
            sb.append("Device: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL).append('\n');
            sb.append("Android: ").append(Build.VERSION.RELEASE).append(" (API ").append(Build.VERSION.SDK_INT).append(")\n");
            sb.append("Date: ").append(new java.util.Date()).append("\n\n=== logcat ===\n");
            try {
                Process p = Runtime.getRuntime().exec(new String[] {"logcat", "-d", "-t", "500"});
                try (BufferedReader br = new BufferedReader(new java.io.InputStreamReader(p.getInputStream()))) {
                    for (String line; (line = br.readLine()) != null; ) sb.append(line).append('\n');
                }
            } catch (Exception e) {
                sb.append("logcat capture error: ").append(e.getMessage()).append('\n');
            }
            File nxLog = new File(dataDir(), "nx.log");
            if (nxLog.canRead()) {
                sb.append("\n=== nx.log").append(nxLog.length() > LOG_TAIL ? " (its last MB)" : "").append(" ===\n");
                sb.append(tail(nxLog, LOG_TAIL));
            }
            File nxLogPrev = new File(dataDir(), "nx.log.prev");
            if (nxLogPrev.canRead()) {
                sb.append("\n=== nx.log.prev").append(nxLogPrev.length() > LOG_TAIL ? " (its last MB)" : "").append(" ===\n");
                sb.append(tail(nxLogPrev, LOG_TAIL));
            }
            try (FileWriter w = new FileWriter(out)) {
                w.write(sb.toString());
            }
            // Listed at once by the file pickers and the Files app, as the saves' export.
            android.media.MediaScannerConnection.scanFile(this, new String[] {out.getPath()}, null, null);
            Toast.makeText(this, Lang.f("Registros exportados a Descargas/%s", "Logs exported to Download/%s",
                    out.getName()), Toast.LENGTH_SHORT).show();
            Uri logUri = Uri.parse("content://" + getPackageName() + ".logprovider" + out.getAbsolutePath());
            Intent share = new Intent(Intent.ACTION_SEND);
            share.setType("text/plain");
            share.putExtra(Intent.EXTRA_STREAM, logUri);
            share.putExtra(Intent.EXTRA_SUBJECT, "NieR Automata debug log");
            share.setClipData(android.content.ClipData.newRawUri("Log", logUri));
            share.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(share, t("Compartir archivo de registros", "Share log file")));
        } catch (Exception e) {
            tell(t("Registros de depuración", "Debug logs"), Lang.f("No se pudieron exportar los registros: %s",
                    "Could not export the logs: %s", String.valueOf(e.getMessage())));
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == PREPARE) {
            refresh();
            return;
        }
        if (result == RESULT_OK && data != null && request == PICK_KEYS) {
            onKeysPicked(data);
            refresh();
            return;
        }
        if (result != RESULT_OK || data == null || data.getData() == null) return;
        if (request == PICK_BASE || request == PICK_UPDATE) {
            onGamePicked(data.getData(), request == PICK_UPDATE);
            refresh();
        } else if (request == PICK_DRIVER && data.getData() != null) importDriver(data.getData());
        else if (request == PICK_ICON) onIconPicked(data.getData());
        else if (request == PICK_MOD) installMod(data.getData());
        else if (request == PICK_ZIP || request == PICK_FOLDER) importSaves(request, data.getData());
    }

    void askAccess() {
        try {
            startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName())));
        } catch (Exception e) {
            startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
        }
    }

    // ---- config.txt: KEY=value lines, in order ----

    static Map<String, String> readConfig(File f) {
        Map<String, String> out = new LinkedHashMap<>();
        if (!f.canRead()) return out;
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            for (String line; (line = r.readLine()) != null; ) {
                line = line.trim();
                int eq = line.indexOf('=');
                if (eq > 0) out.put(line.substring(0, eq), line.substring(eq + 1));
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    static boolean writeConfig(File f, Map<String, String> config) {
        try (FileWriter w = new FileWriter(f)) {
            for (Map.Entry<String, String> e : config.entrySet()) w.write(e.getKey() + "=" + e.getValue() + "\n");
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    void load() {
        config.clear();
        config.putAll(readConfig(new File(dataDir(), "config.txt")));
    }

    void save() {
        if (!hasAccess()) return;
        dataDir().mkdirs();
        if (!dataDir().isDirectory()) return;
        load();
        for (Option o : options) config.put(o.key, o.values[o.pick]);
        final String res = config.get("RESOLUTION");
        if ("360p".equals(res)) {
            config.put("NX_HANDHELD", "1");
            config.put("resolution", "360p");
            config.put("NX_RENDER_RES", "360p");
            config.put("NX_RES", "360p");
        } else if ("480p".equals(res)) {
            config.put("NX_HANDHELD", "1");
            config.put("resolution", "480p");
            config.put("NX_RENDER_RES", "480p");
            config.put("NX_RES", "480p");
        } else if ("1080p".equals(res)) {
            config.put("NX_HANDHELD", "0");
            config.put("resolution", "1080p");
            config.put("NX_RENDER_RES", "1080p");
            config.put("NX_RES", "1080p");
        } else {
            config.put("NX_HANDHELD", "1");
            config.put("resolution", "720p");
            config.put("NX_RENDER_RES", "720p");
            config.put("NX_RES", "720p");
        }
        for (String k : new String[] {"RENDER_SCALE", "NX_ANISO_MAX", "NX_LOD_BIAS"}) config.remove(k);
        final String preset = config.get("GRAPHICS");
        if ("low".equals(preset)) config.put("NX_LOD_BIAS", "1");
        else if ("medium".equals(preset)) config.put("NX_LOD_BIAS", "0.5");
        writeConfig(new File(dataDir(), "config.txt"), config);
    }

    int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    LinearLayout row(String label) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(4), 0, dp(4));
        TextView name = new TextView(this);
        name.setText(label);
        name.setTextColor(Color.WHITE);
        name.setTextSize(17);
        row.addView(name, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        return row;
    }

    Button button(String text, int minWidth, View.OnClickListener click) {
        Button b = new Button(this);
        b.setAllCaps(false);
        b.setMinWidth(dp(minWidth));
        b.setText(text);
        b.setOnClickListener(click);
        return b;
    }

    void addRow(LinearLayout root, String label, String button, View.OnClickListener click) {
        LinearLayout row = row(label);
        row.addView(button(button, 180, click));
        root.addView(row);
    }

    void addPairRow(LinearLayout root, String label, String first, View.OnClickListener firstClick, String second,
            View.OnClickListener secondClick) {
        LinearLayout row = row(label);
        row.addView(button(first, 88, firstClick));
        row.addView(button(second, 88, secondClick));
        root.addView(row);
    }
}
